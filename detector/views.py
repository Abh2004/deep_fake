from django.shortcuts import render, redirect, get_object_or_404
from django.http import Http404
from .forms import VideoUploadForm
from .models import VideoUpload
from .ml_models.deepfake_detector import DeepfakeDetector
import time
import logging

logger = logging.getLogger(__name__)

def home(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_upload = form.save()
            
            try:
                # Process video with deepfake detector
                start_time = time.time()
                detector = DeepfakeDetector()
                logger.info(f"Starting analysis for video: {video_upload.video.name}")
                
                result = detector.predict(video_upload.video.path)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Analysis completed in {elapsed_time:.2f} seconds. Result: {result:.4f}")
                
                # Save result
                video_upload.result = result
                video_upload.processed = True
                video_upload.save()
                
                return redirect('results', video_id=video_upload.id)
            except Exception as e:
                logger.error(f"Error analyzing video: {str(e)}")
                # Set a default result if analysis fails
                video_upload.result = 0.5  # Neutral score
                video_upload.processed = True
                video_upload.save()
                return redirect('results', video_id=video_upload.id)
    else:
        form = VideoUploadForm()
    
    return render(request, 'detector/home.html', {'form': form})

def results(request, video_id):
    try:
        video = get_object_or_404(VideoUpload, id=video_id)
        if not video.processed:
            return render(request, 'detector/processing.html', {'video': video})
        
        # Convert score to percentage and classification
        fake_probability = video.result * 100
        
        # For videos with uncertain results (middle scores), indicate uncertainty
        uncertainty = "low"
        if 40 <= fake_probability <= 60:
            uncertainty = "high"
        elif 30 <= fake_probability <= 70:
            uncertainty = "medium"
        
        context = {
            'video': video,
            'fake_probability': fake_probability,
            'uncertainty': uncertainty
        }
        return render(request, 'detector/results.html', context)
    except Http404:
        return redirect('home')