document.addEventListener("DOMContentLoaded", function () {
    loadHistory();
});

function searchRecipes() {
    const query = document.getElementById('search-input').value.toLowerCase();
    fetch(`/search?query=${query}`)
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data);
        })
        .catch(error => console.error('Error fetching recipes:', error));
}

function displaySearchResults(recipes) {
    const resultsSection = document.getElementById('search-results');
    resultsSection.innerHTML = '';

    recipes.forEach(recipe => {
        const recipeItem = document.createElement('div');
        recipeItem.classList.add('result-item');
        
        const recipeName = document.createElement('span');
        recipeName.textContent = recipe['Recipe Name'] || 'Unnamed Recipe';

        const addButton = document.createElement('button');
        addButton.textContent = 'Add';
        addButton.classList.add('add');
        addButton.onclick = () => addToHistory(recipe['food_id']);

        recipeItem.appendChild(recipeName);
        recipeItem.appendChild(addButton);

        resultsSection.appendChild(recipeItem);
    });
}

function loadHistory() {
    fetch(`/history`)
        .then(response => response.json())
        .then(data => {
            displayHistory(data);
        })
        .catch(error => console.error('Error loading history:', error));
}

function displayHistory(history) {
    const historySection = document.getElementById('history');
    historySection.innerHTML = '';

    if (history.length === 0) {
        historySection.innerHTML = '<p>No history available</p>';
    } else {
        history.forEach(recipe => {
            const historyItem = document.createElement('div');
            historyItem.classList.add('history-item');
            
            const recipeName = document.createElement('span');
            recipeName.textContent = recipe['Recipe Name'] || 'Unnamed Recipe';

            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Delete';
            deleteButton.classList.add('delete');
            deleteButton.onclick = () => deleteFromHistory(recipe['food_id']);

            historyItem.appendChild(recipeName);
            historyItem.appendChild(deleteButton);

            historySection.appendChild(historyItem);
        });
    }
}

function addToHistory(foodId) {
    fetch(`/add_to_history/${foodId}`, { method: 'POST' })
        .then(response => response.json())
        .then(history => {
            displayHistory(history);
        })
        .catch(error => console.error('Error adding to history:', error));
}

function deleteFromHistory(foodId) {
    fetch(`/delete_from_history/${foodId}`, { method: 'DELETE' })
        .then(response => response.json())
        .then(history => {
            displayHistory(history);
        })
        .catch(error => console.error('Error deleting from history:', error));
}

function showRecommendations() {
    fetch(`/recommendations`)
        .then(response => response.json())
        .then(data => {
            alert('Recommendations:\n' + data.map(rec => rec['Recipe Name']).join("\n"));
        })
        .catch(error => console.error('Error fetching recommendations:', error));
}