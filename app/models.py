from django.db import models

class ImageCaption(models.Model):
    image = models.ImageField(upload_to='images/')
    caption = models.TextField(blank=True)
    Tcaption = models.TextField(blank=True)
    language = models.CharField(max_length=20, choices=[('en', 'English'), ('hi', 'Hindi'), ('ta', 'Tamil'), ('te', 'Telugu'), ('fr', 'French'), ('de', 'German'), ('es', 'Spanish'), ('it', 'Italian'), ('ja', 'Japanese'), ('ko', 'Korean')], default='english')

    def __str__(self):
        return self.image.name
