const chatbox = document.getElementById('chatbox');
const input = document.getElementById('input');

function sendMessage() {
    const userMessage = input.value;
    if (!userMessage) return;

    // Afficher le message de l'utilisateur
    chatbox.innerHTML += `<div><strong>Vous:</strong> ${userMessage}</div>`;
    input.value = '';

    // Envoyer la question à l'API Python
    fetch('https://christai.onrender.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            const botResponse = data.response;
            chatbox.innerHTML += `<div><strong>Chatbot:</strong> ${botResponse}</div>`;
        } else {
            chatbox.innerHTML += `<div><strong>Chatbot:</strong> Une erreur est survenue.</div>`;
        }
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll vers le bas
    })
    .catch(error => {
        console.error('Error:', error);
        chatbox.innerHTML += `<div><strong>Chatbot:</strong> Une erreur est survenue.</div>`;
    });
}
// Détecter la touche Entrée dans le champ d'entrée
input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        sendMessage(); // Appeler la fonction sendMessage()
    }
});