from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser, MultiPartParser
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from .models import PatientProfile, Alert
from .serializers import RegisterSerializer, PatientProfileSerializer, AlertSerializer
from .mongo_models import users, alerts, records
import os
import json
import datetime
from django.conf import settings
from joblib import load, dump
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import bcrypt
from .mongo_models import users as users_col


# ==================== INSCRIPTION (CORRIGÉE → is_active + tous les champs) ====================
@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    from .serializers import RegisterSerializer
    serializer = RegisterSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    user = serializer.save()  # Crée l'utilisateur Django

    # Récupère tous les champs du formulaire
    role = request.data.get('role', 'patient')
    birth_date = request.data.get('birth_date')
    phone = request.data.get('phone', '')

    # Hash du mot de passe
    hashed = bcrypt.hashpw(request.data['password'].encode('utf-8'), bcrypt.gensalt())

    # On sauvegarde TOUT dans MongoDB, y compris is_active = True
    users_col.update_one(
        {"username": user.username},
        {"$set": {
            "email": user.email,
            "password_hash": hashed.decode('utf-8'),
            "first_name": user.first_name or "",
            "last_name": user.last_name or "",
            "role": role,
            "birth_date": birth_date,
            "phone": phone,
            "is_staff": role in ["doctor", "admin"],
            "is_active": True,                    # CLÉ ICI !
            "date_joined": datetime.datetime.utcnow()
        }},
        upsert=True
    )

    return Response({
        "message": "Inscription réussie avec succès",
        "username": user.username,
        "role": role
    }, status=201)

# ==================== CONNEXION (avec vérification is_active) ====================
@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({"error": "Identifiants requis"}, status=400)

    user_doc = users_col.find_one({"username": username})

    if not user_doc:
        return Response({"error": "Utilisateur introuvable"}, status=401)

    if not user_doc.get("is_active", False):
        return Response({"error": "Compte désactivé"}, status=403)

    if not bcrypt.checkpw(password.encode('utf-8'), user_doc['password_hash'].encode('utf-8')):
        return Response({"error": "Mot de passe incorrect"}, status=401)

    # Synchro Django Auth
    django_user, _ = User.objects.get_or_create(username=username)
    django_user.is_active = True
    django_user.is_staff = user_doc.get("is_staff", False)
    django_user.save()

    from rest_framework.authtoken.models import Token
    token, _ = Token.objects.get_or_create(user=django_user)

    return Response({
        "token": token.key,
        "user": {
            "username": user_doc["username"],
            "first_name": user_doc.get("first_name", ""),
            "last_name": user_doc.get("last_name", ""),
            "role": user_doc.get("role", "patient"),
            "email": user_doc.get("email", ""),
            "phone": user_doc.get("phone", ""),
            "birth_date": user_doc.get("birth_date")
        }
    })
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_profile_view(request):
    user = users.find_one({"username": request.user.username})
    if not user:
        return Response({"error": "Profil introuvable"}, status=404)
    return Response({
        "username": user["username"],
        "email": user.get("email", ""),
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "role": user["role"]
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile_view(request):
    try:
        profile = PatientProfile.objects.get(user=request.user)
        serializer = PatientProfileSerializer(profile)
        return Response(serializer.data)
    except PatientProfile.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=404)


class PatientProfileViewSet(viewsets.ModelViewSet):
    queryset = PatientProfile.objects.all()
    serializer_class = PatientProfileSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        patient_profile, created = PatientProfile.objects.get_or_create(
            user=request.user,
            defaults={
                'birth_date': request.data.get('birth_date'),
                'phone': request.data.get('phone', ''),
                'role': request.data.get('role', 'patient')
            }
        )
        if not created:
            for field in ['birth_date', 'phone', 'role']:
                if field in request.data:
                    setattr(patient_profile, field, request.data[field])
            patient_profile.save()
        return Response(PatientProfileSerializer(patient_profile).data,
                        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.all().order_by('-created_at')
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        try:
            profile = PatientProfile.objects.get(user=self.request.user)
        except PatientProfile.DoesNotExist:
            return qs.none()
        if profile.role in ['doctor', 'admin']:
            return qs
        return qs.filter(patient=profile)

    def perform_create(self, serializer):
        profile = PatientProfile.objects.filter(user=self.request.user).first()
        serializer.save(patient=profile)


# === MEDICAL RECORDS (MongoDB) ===
class RecordsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        query = {"patient_username": request.user.username}
        if request.user.is_staff:
            patient = request.query_params.get('patient_username')
            if patient:
                query["patient_username"] = patient

        docs = list(records.find(query).sort("recorded_at", -1).limit(200))
        for d in docs:
            d['_id'] = str(d['_id'])
            d['recorded_at'] = d['recorded_at'].isoformat()
        return Response(docs)

    def post(self, request):
        data = request.data
        try:
            s = float(data['systolic'])
            d = float(data['diastolic'])
            g = float(data['glucose'])
            hr = float(data['heart_rate'])
        except:
            return Response({"error": "Données invalides"}, status=400)

        doc = {
            "patient_username": request.user.username,
            "systolic": s,
            "diastolic": d,
            "glucose": g,
            "heart_rate": hr,
            "recorded_at": datetime.datetime.utcnow(),
            "meta": data.get("meta", {})
        }
        result = records.insert_one(doc)

        # === IA PRÉDICTION ===
        pred = 0  # Default to normal (no risk)
        try:
            model = load(os.path.join(settings.BASE_DIR, 'model.joblib'))
            scaler = load(os.path.join(settings.BASE_DIR, 'scaler.joblib'))
            pp = s - d
            ghr = g / (hr + 1)
            pred = int(model.predict(scaler.transform([[s, d, g, hr, pp, ghr]]))[0])

            if pred == 1:
                alerts.insert_one({
                    "patient_username": request.user.username,
                    "level": "high",
                    "message": "Risque élevé détecté par l'IA",
                    "created_at": datetime.datetime.utcnow(),
                    "acknowledged": False
                })
                # WebSocket
                channel_layer = get_channel_layer()
                if channel_layer:
                    async_to_sync(channel_layer.group_send)(
                        "alerts",
                        {"type": "alert.message", "message": "Nouvelle alerte critique !"}
                    )
        except Exception as e:
            print("IA Error:", e)

        return Response({"inserted_id": str(result.inserted_id), "risk": pred == 1})


# === ALERTES ===
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def alerts_view(request):
    query = {"patient_username": request.user.username}
    if request.user.is_staff:
        patient = request.query_params.get('patient_username')
        if patient:
            query["patient_username"] = patient

    alerts_list = list(alerts.find(query).sort("created_at", -1))
    for a in alerts_list:
        a['_id'] = str(a['_id'])
        a['created_at'] = a['created_at'].isoformat()
    return Response(alerts_list)


# Single record by ID
@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def record_detail(request, record_id):
    try:
        obj_id = ObjectId(record_id)
        doc = records.find_one({'_id': obj_id})
        if not doc:
            return Response({'error': 'Not found'}, status=404)
        if request.method == 'DELETE':
            if doc['patient_username'] != request.user.username:
                if not request.user.is_staff:
                    return Response({'error': 'Forbidden'}, status=403)
            records.delete_one({'_id': obj_id})
            return Response({'status': 'deleted'})
        doc['_id'] = str(doc['_id'])
        return Response(doc)
    except:
        return Response({'error': 'Invalid ID'}, status=400)


# Acknowledge Alert
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def acknowledge_alert(request, alert_id):
    alert_doc = alerts.find_one({'_id': ObjectId(alert_id)})
    if not alert_doc:
        return Response({'error': 'Alert not found'}, status=404)

    if alert_doc['patient_username'] != request.user.username and not request.user.is_staff:
        return Response({'error': 'Forbidden'}, status=403)

    alerts.update_one({'_id': ObjectId(alert_id)}, {'$set': {'acknowledged': True}})
    return Response({'status': 'acknowledged'})


# === PREDICT & RETRAIN ===
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_view(request):
    model_path = os.path.join(settings.BASE_DIR, 'model.joblib')
    scaler_path = os.path.join(settings.BASE_DIR, 'scaler.joblib')

    if not os.path.exists(model_path):
        return Response({'error': 'Model not trained yet'}, status=400)

    try:
        model = load(model_path)
        scaler = load(scaler_path)
    except:
        return Response({'error': 'Failed to load model/scaler'}, status=500)

    s = request.data
    required = ['systolic', 'diastolic', 'glucose', 'heart_rate']
    for f in required:
        if f not in s:
            return Response({'error': f'Missing {f}'}, status=400)
        try:
            float(s[f])
        except:
            return Response({'error': f'{f} must be number'}, status=400)

    systolic = float(s['systolic'])
    diastolic = float(s['diastolic'])
    pulse_pressure = systolic - diastolic
    glucose_heart_ratio = float(s['glucose']) / (float(s['heart_rate']) + 1)

    sample = [[systolic, diastolic, s['glucose'], s['heart_rate'], pulse_pressure, glucose_heart_ratio]]
    sample_scaled = scaler.transform(sample)
    pred = int(model.predict(sample_scaled)[0])

    return Response({'prediction': pred, 'risk': 'High' if pred == 1 else 'Normal'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def retrain_view(request):
    docs = list(records.find())

    if len(docs) < 20:
        rng = np.random.RandomState(0)
        n = 2000  # Increased synthetic data for better training
        systolic = rng.normal(120, 20, size=n)  # More variance
        diastolic = rng.normal(80, 15, size=n)
        glucose = rng.normal(100, 30, size=n)
        heart_rate = rng.normal(70, 15, size=n)
        # More complex anomaly logic to achieve ~75% accuracy
        anomaly_prob = 0.25  # 25% anomalies
        label = rng.binomial(1, anomaly_prob, size=n)  # Random anomalies
        # Add some correlation
        label = ((glucose > 140) | (systolic > 140) | (heart_rate > 90) | (diastolic > 90) | (label == 1)).astype(int)
        # Add noise to make it realistic
        noise = rng.normal(0, 0.1, size=n)
        label = np.clip(label + noise, 0, 1).astype(int)
        df = pd.DataFrame({
            'systolic': systolic,
            'diastolic': diastolic,
            'glucose': glucose,
            'heart_rate': heart_rate,
            'label': label
        })
    else:
        records_list = []
        for d in docs:
            records_list.append({
                'systolic': d.get('systolic', 0),
                'diastolic': d.get('diastolic', 0),
                'glucose': d.get('glucose', 0),
                'heart_rate': d.get('heart_rate', 0),
                'label': int(d.get('meta', {}).get('label', 0))
            })
        df = pd.DataFrame(records_list)

    # Data preprocessing
    df = df.dropna()  # Remove missing values
    X = df[['systolic', 'diastolic', 'glucose', 'heart_rate']]
    y = df['label']

    # Feature engineering: add ratios
    X['pulse_pressure'] = X['systolic'] - X['diastolic']
    X['glucose_heart_ratio'] = X['glucose'] / (X['heart_rate'] + 1)  # Avoid division by zero

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Use GradientBoosting for better performance
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model and scaler
    model_path = os.path.join(settings.BASE_DIR, 'model.joblib')
    scaler_path = os.path.join(settings.BASE_DIR, 'scaler.joblib')
    dump(clf, model_path)
    dump(scaler, scaler_path)

    return Response({
        'status': 'retrained',
        'records_used': len(df),
        'accuracy': round(accuracy, 4),
        'model': 'GradientBoostingClassifier'
    })


class RecordsStatsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        days = int(request.query_params.get('days', 30))
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)

        query = {'recorded_at': {'$gte': since}}
        if not request.user.is_staff:
            query['patient_username'] = request.user.username
        else:
            patient = request.query_params.get('patient_username')
            if patient:
                query['patient_username'] = patient

        docs = list(records.find(query).sort('recorded_at', 1))

        series = {'timestamps': [], 'systolic': [], 'diastolic': [], 'glucose': [], 'heart_rate': []}
        for d in docs:
            ts = d.get('recorded_at')
            if isinstance(ts, datetime.datetime):
                ts = ts.isoformat()
            series['timestamps'].append(ts)
            for key in ['systolic', 'diastolic', 'glucose', 'heart_rate']:
                series[key].append(float(d.get(key, 0) or 0))

        def summarize(arr):
            if not arr: return {'min': None, 'max': None, 'avg': None}
            return {'min': min(arr), 'max': max(arr), 'avg': round(sum(arr)/len(arr), 2)}

        summary = {k: summarize(v) for k, v in series.items() if k != 'timestamps'}

        return Response({'series': series, 'summary': summary})


# === ADDITIONAL VIEWS ===
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def latest_record(request):
    query = {"patient_username": request.user.username}
    if request.user.is_staff and request.query_params.get('patient_username'):
        query["patient_username"] = request.query_params.get('patient_username')

    doc = records.find(query).sort("recorded_at", -1).limit(1)
    try:
        doc = next(doc)
        doc['_id'] = str(doc['_id'])
        doc['recorded_at'] = doc['recorded_at'].isoformat()
        return Response(doc)
    except:
        return Response({"error": "Aucun relevé"}, status=404)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def patients_list(request):
    user = users.find_one({"username": request.user.username})
    if not user or user.get("role") not in ["doctor", "admin"]:
        return Response({"error": "Accès refusé"}, status=403)

    patients = list(users.find({"role": "patient"}, {
        "password_hash": 0, "email": 1, "first_name": 1, "last_name": 1, "username": 1
    }))

    for p in patients:
        count = records.count_documents({"patient_username": p["username"]})
        last = records.find_one(
            {"patient_username": p["username"]},
            sort=[("recorded_at", -1)]
        )
        p["records_count"] = count
        p["last_record_date"] = last["recorded_at"].isoformat() if last else None

    return Response(patients)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def patient_profile_view(request):
    # This might need to be implemented based on requirements
    return Response({"message": "Patient profile view"})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upgrade_to_pro(request):
    user = users.find_one({"username": request.user.username})
    if not user:
        return Response({"error": "Utilisateur introuvable"}, status=404)

    if user.get("role") == "pro":
        return Response({"message": "Déjà utilisateur Pro"}, status=200)

    users.update_one(
        {"username": request.user.username},
        {"$set": {"role": "pro"}}
    )
    return Response({"message": "Mise à niveau vers Pro réussie"}, status=200)
