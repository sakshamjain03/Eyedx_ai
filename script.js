const { createApp, ref } = Vue;

createApp({
    setup() {
        const messages = ref([]);
        const userInput = ref("");

        const sendMessage = async () => {
            if (!userInput.value.trim()) return;

            // Push user message
            messages.value.push({ id: Date.now(), text: userInput.value, sender: "user" });

            try {
                const response = await fetch("http://127.0.0.1:8000/generate/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ instruction: userInput.value })
                });

                const result = await response.json();

                messages.value.push({
                    id: Date.now() + 1,
                    text: result.generated_text || "⚠️ Error generating response",
                    sender: "bot"
                });
            } catch (error) {
                messages.value.push({
                    id: Date.now() + 2,
                    text: "❌ Failed to fetch response from backend.",
                    sender: "bot"
                });
            }

            userInput.value = "";
        };

        return { messages, userInput, sendMessage };
    }
}).mount("#app");
