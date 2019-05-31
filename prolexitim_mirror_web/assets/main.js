const videoWidth = 320;
const videoHeight = 240;
const stats = new Stats();

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isiOS();
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('player');
    player.width = videoWidth;
    player.height = videoHeight;

    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}

function setupFPS() {
    stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
    document.getElementById('stats').append(stats.domElement);
}

function toInputTensor(input) {
    return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

function detectEmotions(video, net) {
    const canvas = document.getElementById('player');
    // since images are being fed from a webcam
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const prediction_text = document.getElementById('prediction_text');

    // hack to wait until the canvas is ready
    document.createElement('player');

    const emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy',
        'neutral', 'sad', 'surprised'];

    async function detectEmotionsFrame() {
        stats.begin();

        const imageTensor = toInputTensor(video);

        tf.browser.toPixels(imageTensor, canvas);

        outputs = net.predict(imageTensor);

        const predictions = outputs.dataSync();
        const index_max = predictions.indexOf(Math.max(...predictions));

        prediction_text.innerHTML = emotions[index_max] + " " + predictions[index_max];

        stats.end();
        requestAnimationFrame(detectEmotionsFrame);
    }

    detectEmotionsFrame();
}

async function bindPage() {
    // Load the PoseNet model weights with architecture 0.75
    const net = await tf.loadGraphModel('/assets/model.json');

    document.getElementById('loading-model').style.display = 'none';

    let video;

    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this browser does not support video capture, or this device does not have a camera';
        info.style.display = 'block';
        throw e;
    }

    setupFPS();
    detectEmotions(video, net);
}

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();