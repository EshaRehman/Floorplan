document.getElementById('generate-btn').addEventListener('click', function () {
    const userInput = document.getElementById('user-input').value;
    
    if (userInput.trim() === "") {
        alert("Please enter a description for the floorplan.");
        return;
    }

    // Send POST request to backend
    fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        alert("Floorplan generation request sent successfully!");
        // Redirect or handle UI updates as needed
        console.log(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert("Something went wrong. Please try again.");
    });
});
