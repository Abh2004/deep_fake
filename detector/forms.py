from django import forms
from .models import VideoUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['video']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['video'].widget.attrs.update({
            'class': 'form-control',
            'accept': 'video/*'
        })
        
    def clean_video(self):
        video = self.cleaned_data.get('video')
        if video:
            ext = video.name.split('.')[-1].lower()
            if ext not in ['mp4', 'avi', 'mov', 'mkv']:
                raise forms.ValidationError("Unsupported file format. Use MP4, AVI, MOV, or MKV.")
            if video.size > 100 * 1024 * 1024:  # 100MB
                raise forms.ValidationError("File too large. Maximum size is 100MB.")
        return video