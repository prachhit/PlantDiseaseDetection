from django.contrib import admin

# Register your models here.
from .models import DiseaseInfo # Import the new model

@admin.register(DiseaseInfo)
class DiseaseInfoAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name', 'symptoms')