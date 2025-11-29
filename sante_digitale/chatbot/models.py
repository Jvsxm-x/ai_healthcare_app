from django.db import models
from django.utils import timezone
from authentication.models import User, PatientProfile


class MedicalChat(models.Model):
    """Medical chat conversation stored in MongoDB"""
    
    id = models.CharField(primary_key=True, max_length=100)  # MongoDB ObjectId as string
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name='medical_chats')
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'medical_chat'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['patient', 'created_at']),
            models.Index(fields=['session_id']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        return f"Chat with {self.patient.user.get_full_name()} - {self.session_id}"


class ChatMessage(models.Model):
    """Individual chat messages"""
    
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('bot', 'Bot Response'),
        ('system', 'System Message'),
    ]
    
    id = models.CharField(primary_key=True, max_length=100)  # MongoDB ObjectId as string
    chat = models.ForeignKey(MedicalChat, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # For bot responses, store intent and confidence
    intent = models.CharField(max_length=100, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    
    # For medical context
    medical_context = models.JSONField(blank=True, null=True)
    
    class Meta:
        db_table = 'chat_message'
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['chat', 'timestamp']),
            models.Index(fields=['message_type']),
        ]
    
    def __str__(self):
        return f"{self.get_message_type_display()}: {self.content[:50]}..."


class ChatbotIntent(models.Model):
    """Predefined chatbot intents and responses"""
    
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    keywords = models.JSONField(help_text="List of keywords that trigger this intent")
    response_template = models.TextField()
    is_medical = models.BooleanField(default=False)
    requires_doctor = models.BooleanField(default=False)
    priority = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'chatbot_intent'
        ordering = ['-priority', 'name']
        indexes = [
            models.Index(fields=['is_active', 'priority']),
            models.Index(fields=['is_medical']),
        ]
    
    def __str__(self):
        return self.name


class MedicalKnowledgeBase(models.Model):
    """Medical knowledge base for chatbot responses"""
    
    category = models.CharField(max_length=100)
    topic = models.CharField(max_length=200)
    question = models.TextField()
    answer = models.TextField()
    keywords = models.JSONField()
    confidence_score = models.FloatField(default=0.8)
    source = models.CharField(max_length=200, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'medical_knowledge'
        unique_together = ['category', 'topic']
        indexes = [
            models.Index(fields=['category', 'topic']),
            models.Index(fields=['keywords']),
        ]
    
    def __str__(self):
        return f"{self.category}: {self.topic}"
