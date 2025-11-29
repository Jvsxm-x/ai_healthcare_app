from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Q, F
from django.shortcuts import get_object_or_404
import uuid
import re
from datetime import timedelta

from .models import MedicalChat, ChatMessage, ChatbotIntent, MedicalKnowledgeBase
from .serializers import (
    ChatMessageSerializer, MedicalChatSerializer, ChatCreateSerializer,
    ChatResponseSerializer, ChatHistorySerializer, ChatbotIntentSerializer,
    MedicalKnowledgeBaseSerializer
)
from authentication.models import PatientProfile, DoctorProfile
from authentication.permissions import IsOwner, IsDoctorOrOwner
from notifications.service import notification_service


class IsPatientOrDoctor(permissions.BasePermission):
    """Custom permission to only allow patients or doctors to access chat"""
    
    def has_permission(self, request, view):
        return request.user.is_authenticated and (
            hasattr(request.user, 'patientprofile') or 
            hasattr(request.user, 'doctorprofile') or 
            request.user.is_staff
        )


class MedicalChatViewSet(viewsets.ModelViewSet):
    """ViewSet for medical chat sessions"""
    
    queryset = MedicalChat.objects.all()
    serializer_class = MedicalChatSerializer
    permission_classes = [IsPatientOrDoctor]
    
    def get_queryset(self):
        """Filter chats based on user role"""
        user = self.request.user
        if user.is_staff:
            return MedicalChat.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see chats of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return MedicalChat.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own chats
            return MedicalChat.objects.filter(patient=user.patientprofile)
        return MedicalChat.objects.none()
    
    @action(detail=False, methods=['post'])
    def send_message(self, request):
        """Send a message to the chatbot"""
        serializer = ChatCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = request.user
        if not hasattr(user, 'patientprofile'):
            return Response(
                {'error': 'Only patients can send chat messages'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        patient = user.patientprofile
        message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id')
        
        # Get or create chat session
        if session_id:
            chat = get_object_or_404(MedicalChat, session_id=session_id, patient=patient)
        else:
            chat, created = MedicalChat.objects.get_or_create(
                patient=patient,
                session_id=str(uuid.uuid4()),
                defaults={'is_active': True}
            )
        
        # Save user message
        user_message = ChatMessage.objects.create(
            id=str(uuid.uuid4()),
            chat=chat,
            message_type='user',
            content=message
        )
        
        # Generate bot response
        bot_response = self._generate_bot_response(patient, message)
        
        # Save bot response
        bot_message = ChatMessage.objects.create(
            id=str(uuid.uuid4()),
            chat=chat,
            message_type='bot',
            content=bot_response['response'],
            intent=bot_response.get('intent'),
            confidence=bot_response.get('confidence'),
            medical_context=bot_response.get('medical_context')
        )
        
        # Update chat timestamp
        chat.updated_at = timezone.now()
        chat.save()
        
        # Check if doctor notification is needed
        if bot_response.get('requires_doctor'):
            self._notify_doctor(patient, message, bot_response['response'])
        
        return Response({
            'session_id': chat.session_id,
            'user_message': ChatMessageSerializer(user_message).data,
            'bot_response': ChatMessageSerializer(bot_message).data,
            'requires_doctor': bot_response.get('requires_doctor', False)
        })
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        """Get chat history for a patient"""
        patient_id = request.query_params.get('patient_id')
        session_id = request.query_params.get('session_id')
        
        user = request.user
        
        # Validate access
        if patient_id:
            if hasattr(user, 'patientprofile') and str(user.patientprofile.id) != patient_id:
                return Response({'error': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)
            elif hasattr(user, 'doctorprofile'):
                patient = get_object_or_404(PatientProfile, id=patient_id)
                if patient not in user.doctorprofile.patients.all():
                    return Response({'error': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)
            else:
                return Response({'error': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)
            
            chats = MedicalChat.objects.filter(patient_id=patient_id)
        elif session_id:
            chat = get_object_or_404(MedicalChat, session_id=session_id)
            
            # Check access
            if hasattr(user, 'patientprofile') and chat.patient != user.patientprofile:
                return Response({'error': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)
            elif hasattr(user, 'doctorprofile'):
                if chat.patient not in user.doctorprofile.patients.all():
                    return Response({'error': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)
            
            chats = MedicalChat.objects.filter(session_id=session_id)
        else:
            if hasattr(user, 'patientprofile'):
                chats = MedicalChat.objects.filter(patient=user.patientprofile)
            elif hasattr(user, 'doctorprofile'):
                doctor_patients = user.doctorprofile.patients.all()
                chats = MedicalChat.objects.filter(patient__in=doctor_patients)
            else:
                chats = MedicalChat.objects.none()
        
        # Serialize chats with messages
        result = []
        for chat in chats.order_by('-updated_at'):
            serializer = ChatHistorySerializer({
                'session_id': chat.session_id,
                'messages': chat.messages.all().order_by('timestamp'),
                'created_at': chat.created_at,
                'updated_at': chat.updated_at,
                'message_count': chat.messages.count()
            })
            result.append(serializer.data)
        
        return Response(result)
    
    def _generate_bot_response(self, patient, message):
        """Generate bot response using intent matching and knowledge base"""
        message_lower = message.lower()
        
        # Check for medical emergency keywords
        emergency_keywords = ['urgence', 'emergency', 'douleur', 'pain', 'aide', 'help']
        if any(keyword in message_lower for keyword in emergency_keywords):
            return {
                'response': "Je comprends que vous avez besoin d'aide immédiate. Si c'est une urgence médicale, veuillez contacter les services d'urgence au 15 ou vous rendre aux urgences les plus proches. Souhaitez-vous que je prévienne votre médecin traitant ?",
                'intent': 'emergency',
                'confidence': 0.9,
                'requires_doctor': True,
                'medical_context': {'emergency': True}
            }
        
        # Check intents
        intents = ChatbotIntent.objects.filter(is_active=True).order_by('-priority')
        for intent in intents:
            if any(keyword.lower() in message_lower for keyword in intent.keywords):
                response = intent.response_template
                
                # Personalize response
                if '{name}' in response:
                    response = response.replace('{name}', patient.user.get_full_name())
                
                return {
                    'response': response,
                    'intent': intent.name,
                    'confidence': 0.8,
                    'requires_doctor': intent.requires_doctor,
                    'medical_context': {'intent': intent.name, 'is_medical': intent.is_medical}
                }
        
        # Search knowledge base
        knowledge = MedicalKnowledgeBase.objects.filter(
            keywords__contains=[message_lower]
        ).order_by('-confidence_score').first()
        
        if knowledge:
            return {
                'response': knowledge.answer,
                'intent': 'knowledge_base',
                'confidence': knowledge.confidence_score,
                'requires_doctor': False,
                'medical_context': {'category': knowledge.category, 'topic': knowledge.topic}
            }
        
        # Default responses based on common patterns
        if 'bonjour' in message_lower or 'salut' in message_lower:
            return {
                'response': f"Bonjour {patient.user.get_full_name()} ! Comment puis-je vous aider aujourd'hui concernant votre santé ?",
                'intent': 'greeting',
                'confidence': 0.9
            }
        
        if 'merci' in message_lower:
            return {
                'response': "Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur votre santé.",
                'intent': 'thanks',
                'confidence': 0.9
            }
        
        # Check for measurement-related questions
        if any(word in message_lower for word in ['tension', 'pression', 'glycémie', 'poids', 'taille']):
            return {
                'response': "Pour consulter vos dernières mesures, vous pouvez utiliser l'application mobile ou me demander spécifiquement 'mes dernières mesures'. Si vous avez des questions sur des valeurs anormales, je peux vous aider à les comprendre.",
                'intent': 'measurements',
                'confidence': 0.7,
                'medical_context': {'topic': 'measurements'}
            }
        
        # Default fallback response
        return {
            'response': "Je suis là pour vous aider avec des questions générales sur votre santé et le suivi de vos mesures. Pour des questions médicales spécifiques, je vous recommande de consulter votre médecin traitant. Comment puis-je vous assister aujourd'hui ?",
            'intent': 'fallback',
            'confidence': 0.5
        }
    
    def _notify_doctor(self, patient, user_message, bot_response):
        """Notify doctor about concerning chat message"""
        if patient.assigned_doctor:
            notification_data = {
                'message': f"Message préoccupant du patient {patient.user.get_full_name()}: '{user_message[:50]}...' Réponse: '{bot_response[:50]}...'",
                'type': 'chat_concern'
            }
            notification_service.send_critical_alert(patient, notification_data)


class ChatbotIntentViewSet(viewsets.ModelViewSet):
    """ViewSet for managing chatbot intents (admin only)"""
    
    queryset = ChatbotIntent.objects.all()
    serializer_class = ChatbotIntentSerializer
    permission_classes = [permissions.IsAdminUser]


class MedicalKnowledgeBaseViewSet(viewsets.ModelViewSet):
    """ViewSet for managing medical knowledge base (admin only)"""
    
    queryset = MedicalKnowledgeBase.objects.all()
    serializer_class = MedicalKnowledgeBaseSerializer
    permission_classes = [permissions.IsAdminUser]
