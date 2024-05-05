from django import forms
from .models import ImageCaption

class ImageCaptionForm(forms.ModelForm):
    class Meta:
        model = ImageCaption
        fields = ['image', 'language']
        widgets = {
            'image': forms.ClearableFileInput(attrs={'class': 'form-control mx-auto'}),
            'language': forms.Select(attrs={'class': 'form-control form-select-lg mb-3'})  # Add Bootstrap classes for the language dropdown
        }