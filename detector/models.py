from django.db import models
import os

class VideoUpload(models.Model):
    video = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result = models.FloatField(null=True, blank=True)
    processed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Video: {os.path.basename(self.video.name)}"
        
    def delete(self, *args, **kwargs):
        self.video.delete()
        super().delete(*args, **kwargs)