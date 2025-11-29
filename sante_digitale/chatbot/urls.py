from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import MedicalChatViewSet, ChatbotIntentViewSet, MedicalKnowledgeBaseViewSet

router = DefaultRouter()
router.register(r'chats', MedicalChatViewSet, basename='medical-chat')
router.register(r'intents', ChatbotIntentViewSet, basename='chatbot-intent')
router.register(r'knowledge', MedicalKnowledgeBaseViewSet, basename='medical-knowledge')

urlpatterns = [
    path('', include(router.urls)),
]
