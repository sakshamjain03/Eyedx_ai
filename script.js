document.addEventListener("DOMContentLoaded", function () {
    const chatApp = Vue.createApp({
        data() {
            return {
                messages: [],
                userInput: ""
            };
        },
        methods: {
            async sendMessage() {
                if (this.userInput.trim() === "") return;
                this.messages.push({ text: this.userInput, sender: "user" });

                try {
                    const response = await fetch("http://127.0.0.1:5000/generate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ instruction: this.userInput })
                    });

                    const data = await response.json();
                    this.messages.push({ text: data.generated_text, sender: "bot" });
                } catch (error) {
                    console.error("Error:", error);
                    this.messages.push({ text: "Error fetching response!", sender: "bot" });
                }

                this.userInput = "";
            }
        }
    });

    chatApp.mount("#app");

    // Toggle Chatbox
    document.getElementById("toggle-btn").addEventListener("click", function () {
        const chatbox = document.getElementById("app");
        chatbox.style.display = chatbox.style.display === "none" ? "block" : "none";
    });

    // Camera Access & Capture
    let stream;
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const cameraBtn = document.getElementById("camera-btn");
    const capturedImage = document.getElementById("captured-image");
    const retakeBtn = document.getElementById("retake-btn");
    const submitBtn = document.getElementById("submit-btn");

    cameraBtn.addEventListener("click", function () {
        if (cameraBtn.textContent === "Capture Image") {
            captureImage();
        } else {
            accessCamera();
        }
    });

    function accessCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((mediaStream) => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.style.display = "block";
                cameraBtn.textContent = "Capture Image";  // Change button text
            })
            .catch(err => alert("Camera Access Denied: " + err.message));
    }

    function captureImage() {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageDataUrl = canvas.toDataURL("image/png");

        capturedImage.src = imageDataUrl;
        capturedImage.style.display = "block";
        retakeBtn.style.display = "block";
        submitBtn.style.display = "block";

        video.style.display = "none";
        cameraBtn.style.display = "none";  // Hide camera button after capture

        stream.getTracks().forEach(track => track.stop());
    }

    retakeBtn.addEventListener("click", function () {
        capturedImage.style.display = "none";
        retakeBtn.style.display = "none";
        submitBtn.style.display = "none";
        cameraBtn.style.display = "block"; // Show camera button again
        cameraBtn.textContent = "Access Camera";
        accessCamera();
    });

    submitBtn.addEventListener("click", function () {
        alert("Image submitted!");
    });
});
