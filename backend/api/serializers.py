from rest_framework import serializers
from django.contrib.auth.models import User
from .models import PatientProfile, Alert

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id','username','email']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    role = serializers.CharField(write_only=True, required=False)
    class Meta:
        model = User
        fields = ('username','email','password','role')
    def create(self, validated_data):
        role = validated_data.pop('role', 'patient')
        user = User(username=validated_data['username'], email=validated_data.get('email',''))
        user.set_password(validated_data['password'])
        user.save()
        PatientProfile.objects.create(user=user, role=role)
        return user

class PatientProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    class Meta:
        model = PatientProfile
        fields = '__all__'
# api/serializers.py

class PatientProfileSerializer1(serializers.ModelSerializer):
    first_name = serializers.CharField(source='user.first_name', required=False)
    last_name = serializers.CharField(source='user.last_name', required=False)
    email = serializers.EmailField(source='user.email', read_only=True)
    username = serializers.CharField(source='user.username', read_only=True)
    full_name = serializers.SerializerMethodField()
    date_joined = serializers.DateTimeField(source='user.date_joined', read_only=True, format='%d/%m/%Y')

    class Meta:
        model = PatientProfile
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 'full_name',
            'birth_date', 'phone', 'role', 'date_joined'
        ]
        read_only_fields = ['role', 'email', 'username']

    def get_full_name(self, obj):
        return f"{obj.user.first_name} {obj.user.last_name}".strip() or obj.user.username
class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = '__all__'
