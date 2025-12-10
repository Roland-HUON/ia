const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let session = null;

// Configuration du canvas
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#FFF'; // Dessiner en blanc
ort.env.logLevel = 'error';

// Remplir le canvas en noir au départ (comme MNIST)
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Charger le modèle ONNX au démarrage
async function loadModel() {
    try {
        resultDiv.textContent = '⏳ Chargement du modèle...';
        resultDiv.className = 'result-show';
        
        session = await ort.InferenceSession.create('model.onnx');
        
        resultDiv.textContent = '✅ Modèle chargé ! Dessinez un chiffre';
        setTimeout(() => {
            resultDiv.className = 'result-hidden';
        }, 2000);
    } catch (error) {
        console.error('Erreur lors du chargement du modèle:', error);
        resultDiv.textContent = '❌ Erreur: Modèle non trouvé. Assurez-vous que model.onnx existe dans le dossier.';
        resultDiv.className = 'result-show';
    }
}

// Prétraiter l'image du canvas pour le modèle avec détection de patterns
function preprocessCanvas() {
    // Étape 1 : Détecter la bounding box du chiffre
    const boundingBox = detectBoundingBox();
    
    if (!boundingBox) {
        console.warn('Aucun chiffre détecté');
        return new Float32Array(1 * 1 * 28 * 28).fill(0);
    }
    
    console.log('📦 Bounding box:', boundingBox);
    
    // Étape 2 : Extraire et centrer le chiffre
    const centeredCanvas = extractAndCenterDigit(boundingBox);
    
    // Étape 3 : Redimensionner à 28x28
    const resizedCanvas = document.createElement('canvas');
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    
    // Calculer les dimensions pour conserver le ratio (comme MNIST)
    const size = Math.max(boundingBox.width, boundingBox.height);
    const scale = 20 / size; // MNIST a environ 20px pour le chiffre dans 28x28
    const scaledWidth = boundingBox.width * scale;
    const scaledHeight = boundingBox.height * scale;
    
    // Centrer dans 28x28
    const offsetX = (28 - scaledWidth) / 2;
    const offsetY = (28 - scaledHeight) / 2;
    
    resizedCtx.drawImage(
        centeredCanvas,
        offsetX, offsetY,
        scaledWidth, scaledHeight
    );
    
    // Étape 4 : Convertir en tableau normalisé
    const imageData = resizedCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    
    const input = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const pixelIndex = i * 4;
        const r = pixels[pixelIndex];
        const g = pixels[pixelIndex + 1];
        const b = pixels[pixelIndex + 2];
        
        const gray = (r + g + b) / 3;
        input[i] = gray / 255.0;
    }
    
    visualizePreprocessed(resizedCanvas);
    
    return input;
}

// Détecter la bounding box du chiffre dessiné
function detectBoundingBox() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;
    let hasPixel = false;
    
    // Parcourir tous les pixels pour trouver les limites
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4;
            const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
            
            // Si le pixel est suffisamment lumineux (blanc)
            if (brightness > 30) {
                hasPixel = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    
    if (!hasPixel) return null;
    
    // Ajouter une petite marge
    const margin = 10;
    minX = Math.max(0, minX - margin);
    minY = Math.max(0, minY - margin);
    maxX = Math.min(canvas.width, maxX + margin);
    maxY = Math.min(canvas.height, maxY + margin);
    
    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY
    };
}

// Extraire et centrer le chiffre
function extractAndCenterDigit(bbox) {
    const extractCanvas = document.createElement('canvas');
    const extractCtx = extractCanvas.getContext('2d');
    extractCanvas.width = bbox.width;
    extractCanvas.height = bbox.height;
    
    // Copier la région du chiffre
    extractCtx.drawImage(
        canvas,
        bbox.x, bbox.y, bbox.width, bbox.height,
        0, 0, bbox.width, bbox.height
    );
    
    return extractCanvas;
}

// Fonction pour visualiser l'image prétraitée
function visualizePreprocessed(tempCanvas) {
    const previewContainer = document.getElementById('previewContainer');
    const previewCanvas = document.getElementById('previewCanvas');
    const previewCtx = previewCanvas.getContext('2d');
    
    // Afficher le conteneur
    previewContainer.style.display = 'block';
    
    // Effacer le canvas
    previewCtx.fillStyle = '#000';
    previewCtx.fillRect(0, 0, 140, 140);
    
    // Dessiner l'image 28x28 agrandie 5x pour voir les pixels
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(tempCanvas, 0, 0, 140, 140);
}

// Prédiction avec ONNX Runtime
async function predictDigit() {
    if (!session) {
        resultDiv.textContent = '❌ Modèle non chargé. Veuillez rafraîchir la page.';
        resultDiv.className = 'result-show';
        return;
    }
    
    try {
        const inputData = preprocessCanvas();
        
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        
        const output = results[session.outputNames[0]].data;
        console.log('Sorties complètes du modèle:');
        for (let i = 0; i < output.length; i++) {
            console.log("Digits " + i + " output: " + output[i]);
        }
                
        let maxProb = -Infinity;
        let predictedDigit = -1;
        
        for (let i = 0; i < output.length; i++) {
            if (output[i] > maxProb) {
                maxProb = output[i];
                predictedDigit = i;
            }
        }
        
        console.log("Chiffre prédit:", predictedDigit, "avec valeur:", maxProb);
        
        const expSum = Array.from(output).reduce((sum, val) => sum + Math.exp(val), 0);
        console.log("expSum: " + expSum);
        const confidence = (Math.exp(maxProb) / expSum * 100).toFixed(1);
        console.log("confidence: " + confidence);
        resultDiv.textContent = `🎯 Prédiction : ${predictedDigit} (Confiance : ${confidence}%)`;
        resultDiv.className = 'result-show';
        
    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        resultDiv.textContent = `❌ Erreur: ${error.message}`;
        resultDiv.className = 'result-show';
    }
}

// Initialiser le modèle au chargement de la page
loadModel();

// Fonction pour obtenir la position de la souris
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

// Fonction pour obtenir la position du toucher
function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
    };
}

// Événements souris
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const pos = getMousePos(e);
    lastX = pos.x;
    lastY = pos.y;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    lastX = pos.x;
    lastY = pos.y;
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

canvas.addEventListener('mouseout', () => {
    isDrawing = false;
});

// Événements tactiles (mobile)
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const pos = getTouchPos(e);
    lastX = pos.x;
    lastY = pos.y;
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    
    const pos = getTouchPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    lastX = pos.x;
    lastY = pos.y;
});

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    isDrawing = false;
});

// Bouton Effacer
clearBtn.addEventListener('click', () => {
    // Remettre le fond noir
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Cacher le preview
    const previewContainer = document.getElementById('previewContainer');
    previewContainer.style.display = 'none';
    
    resultDiv.textContent = 'Canvas effacé ! Dessinez un nouveau chiffre';
    resultDiv.className = 'result-show';
    
    setTimeout(() => {
        resultDiv.className = 'result-hidden';
    }, 2000);
});

// Bouton Prédire
predictBtn.addEventListener('click', async () => {
    // Vérifier si le canvas est vide
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    let isEmpty = true;
    
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3] > 0) { // Alpha channel
            isEmpty = false;
            break;
        }
    }
    
    if (isEmpty) {
        resultDiv.textContent = '⚠️ Veuillez dessiner un chiffre d\'abord !';
        resultDiv.className = 'result-show';
        return;
    }
    
    // Faire la prédiction avec ONNX Runtime
    await predictDigit();
});

// Le message initial est géré par loadModel()