const fileInput = document.getElementById('fileInput');
const cameraBtn = document.getElementById('cameraBtn');
const cameraContainer = document.getElementById('cameraContainer');
const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const displayCanvas = document.getElementById('displayCanvas');
const displayCtx = displayCanvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const previewContainer = document.getElementById('previewContainer');
const previewMain = document.getElementById('previewMain');

let session = null;
let currentImage = null;
let stream = null;

// Classes CIFAR-10
const CIFAR10_CLASSES = [
    '✈️ Avion',
    '🚗 Automobile', 
    '🐦 Oiseau',
    '🐱 Chat',
    '🦌 Cerf',
    '🐕 Chien',
    '🐸 Grenouille',
    '🐴 Cheval',
    '🚢 Bateau',
    '🚚 Camion'
];

ort.env.logLevel = 'error';

// Charger le modèle ONNX
async function loadModel() {
    try {
        resultDiv.textContent = '⏳ Chargement du modèle CIFAR-10...';
        resultDiv.className = 'result-show';
        
        session = await ort.InferenceSession.create('model.onnx');
        
        resultDiv.textContent = '✅ Modèle chargé ! Téléchargez une image';
        setTimeout(() => {
            resultDiv.className = 'result-hidden';
        }, 2000);
    } catch (error) {
        console.error('Erreur chargement modèle:', error);
        resultDiv.textContent = '❌ Modèle introuvable. Exportez model.onnx depuis Jupyter.';
        resultDiv.className = 'result-show error';
    }
}

// Gérer l'upload d'image
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            displayImage(img);
            predictBtn.style.display = 'block';
            previewContainer.style.display = 'none';
            resultDiv.textContent = 'Image chargée ! Cliquez sur Analyser';
            resultDiv.className = 'result-show';
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

// Afficher l'image sur le canvas
function displayImage(img) {
    previewMain.style.display = 'block';
    
    // Calculer les dimensions pour garder l'aspect ratio
    const maxSize = 280;
    let width = img.width;
    let height = img.height;
    
    if (width > height) {
        if (width > maxSize) {
            height = (height * maxSize) / width;
            width = maxSize;
        }
    } else {
        if (height > maxSize) {
            width = (width * maxSize) / height;
            height = maxSize;
        }
    }
    
    // Centrer l'image
    const x = (displayCanvas.width - width) / 2;
    const y = (displayCanvas.height - height) / 2;
    
    displayCtx.fillStyle = '#FFF';
    displayCtx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);
    displayCtx.drawImage(img, x, y, width, height);
}

// Gérer la webcam
cameraBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
        cameraContainer.style.display = 'block';
        previewMain.style.display = 'none';
    } catch (error) {
        alert('❌ Impossible d\'accéder à la caméra : ' + error.message);
    }
});

// Capturer une photo
captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const img = new Image();
    img.onload = () => {
        currentImage = img;
        displayImage(img);
        closeCamera();
        predictBtn.style.display = 'block';
        previewContainer.style.display = 'none';
        resultDiv.textContent = 'Photo capturée ! Cliquez sur Analyser';
        resultDiv.className = 'result-show';
    };
    img.src = canvas.toDataURL();
});

// Fermer la caméra
closeCameraBtn.addEventListener('click', closeCamera);

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraContainer.style.display = 'none';
}

// Prétraiter l'image pour CIFAR-10 (224x224 RGB, normalisation ImageNet)
function preprocessImage() {
    if (!currentImage) return null;
    
    // Canvas temporaire 224x224
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 224;
    tempCanvas.height = 224;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Redimensionner l'image vers 224x224
    tempCtx.drawImage(currentImage, 0, 0, 224, 224);
    
    // Afficher l'aperçu
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(tempCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
    previewContainer.style.display = 'block';
    
    // Obtenir les pixels
    const imageData = tempCtx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    
    // Préparer le tensor [1, 3, 224, 224] avec normalisation ImageNet
    const input = new Float32Array(1 * 3 * 224 * 224);
    
    // Normalisation CIFAR-10 (doit correspondre à celle du training)
    const mean = [0.4914, 0.4822, 0.4465];
    const std = [0.2470, 0.2435, 0.2616];
    
    // Convertir RGBA → RGB normalisé (CHW format)
    for (let i = 0; i < 224; i++) {
        for (let j = 0; j < 224; j++) {
            const idx = (i * 224 + j) * 4;
            const baseIdx = i * 224 + j;
            
            // Canal R
            input[baseIdx] = (data[idx] / 255.0 - mean[0]) / std[0];
            // Canal G
            input[224 * 224 + baseIdx] = (data[idx + 1] / 255.0 - mean[1]) / std[1];
            // Canal B  
            input[2 * 224 * 224 + baseIdx] = (data[idx + 2] / 255.0 - mean[2]) / std[2];
        }
    }
    
    return input;
}

// Fonction de prédiction
async function predict() {
    if (!session) {
        resultDiv.textContent = '❌ Modèle non chargé';
        resultDiv.className = 'result-show error';
        return;
    }
    
    if (!currentImage) {
        resultDiv.textContent = '❌ Aucune image chargée';
        resultDiv.className = 'result-show error';
        return;
    }
    
    try {
        resultDiv.textContent = '🔄 Analyse en cours...';
        resultDiv.className = 'result-show';
        
        // Prétraiter l'image
        const inputData = preprocessImage();
        if (!inputData) {
            throw new Error('Erreur de prétraitement');
        }
        
        // Créer le tensor d'entrée
        const tensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
        
        // Exécuter l'inférence
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Obtenir les prédictions
        const output = results.output.data;
        
        // Trouver la classe avec la plus haute probabilité
        let maxIdx = 0;
        let maxVal = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }
        
        // Calculer softmax pour les probabilités
        const expScores = Array.from(output).map(x => Math.exp(x));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probabilities = expScores.map(x => x / sumExp);
        
        // Top 3 prédictions
        const top3 = probabilities
            .map((prob, idx) => ({ class: CIFAR10_CLASSES[idx], prob: prob }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 3);
        
        // Afficher le résultat
        const confidence = (top3[0].prob * 100).toFixed(1);
        
        let resultHTML = `
            <div class="prediction-main">
                <strong>🎯 Prédiction :</strong> ${top3[0].class}<br>
                <strong>📊 Confiance :</strong> ${confidence}%
            </div>
            <div class="prediction-top3">
                <strong>Top 3 :</strong><br>
                1️⃣ ${top3[0].class} - ${(top3[0].prob * 100).toFixed(1)}%<br>
                2️⃣ ${top3[1].class} - ${(top3[1].prob * 100).toFixed(1)}%<br>
                3️⃣ ${top3[2].class} - ${(top3[2].prob * 100).toFixed(1)}%
            </div>
        `;
        
        resultDiv.innerHTML = resultHTML;
        resultDiv.className = 'result-show success';
        
        console.log('Prédictions détaillées:', top3);
        
    } catch (error) {
        console.error('Erreur prédiction:', error);
        resultDiv.textContent = `❌ Erreur: ${error.message}`;
        resultDiv.className = 'result-show error';
    }
}

// Event listeners
predictBtn.addEventListener('click', predict);

// Charger le modèle au démarrage
loadModel();
