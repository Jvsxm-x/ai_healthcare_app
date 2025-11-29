from rest_framework import serializers
from django.utils import timezone
import uuid
from .models import MedicalChat, ChatMessage, ChatbotIntent, MedicalKnowledgeBase


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
    
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'message_type', 'content', 'timestamp',
            'intent', 'confidence', 'medical_context'
        ]
        read_only_fields = ['id', 'timestamp']


class MedicalChatSerializer(serializers.ModelSerializer):
    """Serializer for medical chat sessions"""
    
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = MedicalChat
        fields = [
            'id', 'patient', 'session_id', 'created_at', 'updated_at',
            'is_active', 'messages', 'message_count'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()


class ChatCreateSerializer(serializers.Serializer):
    """Serializer for creating new chat messages"""
    
    message = serializers.CharField(max_length=1000)
    session_id = serializers.CharField(max_length=100, required=False)
    
    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chatbot responses"""
    
    response = serializers.CharField()
    intent = serializers.CharField(required=False)
    confidence = serializers.FloatField(required=False)
    medical_context = serializers.JSONField(required=False)
    requires_doctor = serializers.BooleanField(required=False)


class ChatHistorySerializer(serializers.Serializer):
    """Serializer for chat history"""
    
    session_id = serializers.CharField()
    messages = ChatMessageSerializer(many=True)
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()
    message_count = serializers.IntegerField()


class ChatbotIntentSerializer(serializers.ModelSerializer):
    """Serializer for chatbot intents"""
    
    class Meta:
        model = ChatbotIntent
        fields = [
            'id', 'name', 'description', 'keywords', 'response_template',
            'is_medical', 'requires_doctor', 'priority', 'is_active',
            'created_at', 'updated_at'
        ]


class MedicalKnowledgeBaseSerializer(serializers.ModelSerializer):
    """Serializer for medical knowledge base"""
    
    class Meta:
        model = MedicalKnowledgeBase
        fields = [
            'id', 'category', 'topic', 'question', 'answer',
            'keywords', 'confidence_score', 'source', 'last_updated'
        ]
