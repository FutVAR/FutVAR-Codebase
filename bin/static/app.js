document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('videoFileInput');
    const modelSelect = document.getElementById('modelSelect');
    const inferenceButton = document.getElementById('inferenceButton');
    const resultsArea = document.getElementById('resultsArea');
    const resultsText = document.getElementById('resultsText');
    const videoResults = document.getElementById('videoResults');
    const processedVideoPlayer = document.getElementById('processedVideoPlayer');
    const errorArea = document.getElementById('errorArea');

    // Enable inference button only when both video and model are selected
    [videoInput, modelSelect].forEach(element => {
        element.addEventListener('change', () => {
            const videoSelected = videoInput.files.length > 0;
            const modelSelected = modelSelect.value !== '';
            inferenceButton.disabled = !(videoSelected && modelSelected);
        });
    });

    inferenceButton.addEventListener('click', async () => {
        console.log("Run Inference button clicked!"); // <--- ADD THIS LINE
        resultsArea.classList.remove('hidden');
        resultsText.textContent = 'Processing video...';
        errorArea.classList.add('hidden');
        videoResults.classList.add('hidden');


        const selectedVideoFile = videoInput.files[0];
        const selectedModel = modelSelect.value;

        const formData = new FormData();
        formData.append('videoFile', selectedVideoFile);
        formData.append('model', selectedModel);

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    resultsText.textContent = 'Inference successful!';
                    processedVideoPlayer.src = data.video_url;
                    videoResults.classList.remove('hidden'); // Show video player
                } else {
                    resultsText.textContent = 'Inference failed.';
                    errorArea.textContent = data.error || 'Unknown error during inference.';
                    errorArea.classList.remove('hidden');
                }
            } else {
                resultsText.textContent = 'Error communicating with server.';
                errorArea.textContent = `HTTP error! status: ${response.status}`;
                errorArea.classList.remove('hidden');
            }
        } catch (error) {
            resultsText.textContent = 'Error during processing.';
            errorArea.textContent = error.message || 'An unexpected error occurred.';
            errorArea.classList.remove('hidden');
        }
    });
});