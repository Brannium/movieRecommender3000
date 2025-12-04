// Global state
let state = {
    currentPage: 1,
    limit: 50,
    searchQuery: '',
    totalPages: 1,
    totalMovies: 0,
    ratedMovies: {},  // {movieId: rating}
    userId: 1,  // Default user ID
};

// DOM elements
const searchInput = document.getElementById('searchInput');
const clearBtn = document.getElementById('clearBtn');
const moviesTable = document.getElementById('moviesTable');
const moviesBody = document.getElementById('moviesBody');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const pageInfo = document.getElementById('pageInfo');
const movieCount = document.getElementById('movieCount');
const ratingCount = document.getElementById('ratingCount');
const footerMsg = document.getElementById('footerMsg');
const recommendBtn = document.getElementById('recommendBtn');
const recommendationsModal = document.getElementById('recommendationsModal');
const recommendationsContainer = document.getElementById('recommendationsContainer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadMovies();
    setupEventListeners();
});

function setupEventListeners() {
    searchInput.addEventListener('input', debounce(handleSearch, 300));
    clearBtn.addEventListener('click', clearSearch);
    prevBtn.addEventListener('click', previousPage);
    nextBtn.addEventListener('click', nextPage);
    recommendBtn.addEventListener('click', fetchRecommendations);
}

function debounce(func, delay) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), delay);
    };
}

async function loadMovies() {
    try {
        const url = `/api/movies?page=${state.currentPage}&limit=${state.limit}&search=${encodeURIComponent(state.searchQuery)}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch movies');
        
        const data = await response.json();
        
        state.totalPages = data.total_pages;
        state.totalMovies = data.total;
        
        renderMovies(data.movies);
        updatePaginationControls();
        updateInfoBar(data);
        
    } catch (error) {
        console.error('Error loading movies:', error);
        showError('Failed to load movies. Please try again.');
    }
}

function renderMovies(movies) {
    moviesBody.innerHTML = '';
    
    if (movies.length === 0) {
        moviesBody.innerHTML = '<tr class="loading"><td colspan="2">No movies found.</td></tr>';
        return;
    }
    
    movies.forEach(movie => {
        const row = createMovieRow(movie);
        moviesBody.appendChild(row);
    });
}

function createMovieRow(movie) {
    const row = document.createElement('tr');
    const movieId = movie.id;
    const rating = state.ratedMovies[movieId];
    
    row.innerHTML = `
        <td class="movie-title">${escapeHtml(movie.title)}</td>
        <td>
            <div class="rating-buttons">
                <button class="btn-rating btn-dislike ${rating === -1 ? 'active-dislike' : ''}" 
                        onclick="rateMovie(${movieId}, -1)">👎 Dislike</button>
                <button class="btn-rating btn-like ${rating === 1 ? 'active-like' : ''}" 
                        onclick="rateMovie(${movieId}, 1)">👍 Like</button>
                <button class="btn-rating btn-superlike ${rating === 2 ? 'active-superlike' : ''}" 
                        onclick="rateMovie(${movieId}, 2)">⭐ SuperLike</button>
            </div>
        </td>
    `;
    
    return row;
}

function rateMovie(movieId, rating) {
    // Check if this movie is already rated with this rating
    const currentRating = state.ratedMovies[movieId];
    if (currentRating === rating) {
        // Toggle off: remove the rating
        delete state.ratedMovies[movieId];
        showSuccess('Rating removed!');
    } else {
        // Update local state only
        state.ratedMovies[movieId] = rating;
        showSuccess(`Movie rated! (${getRatingLabel(rating)})`);
    }
    
    // Update UI immediately
    updateMovieRow(movieId);
    updateRatingCount(Object.keys(state.ratedMovies).length);
}

function updateMovieRow(movieId) {
    // Find the specific row for this movie and update only its buttons
    const rows = moviesBody.querySelectorAll('tr');
    rows.forEach(row => {
        const titleCell = row.querySelector('.movie-title');
        if (!titleCell) return;
        
        // Get the buttons in this specific row
        const buttons = row.querySelectorAll('.btn-rating');
        buttons.forEach(btn => {
            btn.classList.remove('active-dislike', 'active-like', 'active-superlike');
        });
        
        // Get the rating for this specific movie from the title cell's parent row
        // We need to match the movieId with the row - use data attribute or parse from onclick
        const rateButtons = row.querySelectorAll('[onclick*="rateMovie"]');
        if (rateButtons.length > 0) {
            // Extract movieId from the first button's onclick attribute
            const firstButtonOnclick = rateButtons[0].getAttribute('onclick');
            const match = firstButtonOnclick.match(/rateMovie\((\d+),/);
            if (match) {
                const rowMovieId = parseInt(match[1]);
                const rating = state.ratedMovies[rowMovieId];
                
                if (rating === -1) {
                    buttons[0].classList.add('active-dislike');
                } else if (rating === 1) {
                    buttons[1].classList.add('active-like');
                } else if (rating === 2) {
                    buttons[2].classList.add('active-superlike');
                }
            }
        }
    });
}

function getRatingLabel(rating) {
    if (rating === -1) return 'Dislike';
    if (rating === 1) return 'Like';
    if (rating === 2) return 'SuperLike';
    return 'Unknown';
}

function handleSearch() {
    state.searchQuery = searchInput.value.trim();
    state.currentPage = 1;  // Reset to first page on search
    loadMovies();
}

function clearSearch() {
    searchInput.value = '';
    state.searchQuery = '';
    state.currentPage = 1;
    searchInput.focus();
    loadMovies();
}

function previousPage() {
    if (state.currentPage > 1) {
        state.currentPage--;
        loadMovies();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function nextPage() {
    if (state.currentPage < state.totalPages) {
        state.currentPage++;
        loadMovies();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function updatePaginationControls() {
    prevBtn.disabled = state.currentPage <= 1;
    nextBtn.disabled = state.currentPage >= state.totalPages;
    pageInfo.textContent = `Page ${state.currentPage} of ${state.totalPages}`;
}

function updateInfoBar(data) {
    const searchTerm = state.searchQuery ? ` (searching for "${state.searchQuery}")` : '';
    movieCount.textContent = `Showing ${data.movies.length} of ${data.total} movies${searchTerm}`;
}

function updateRatingCount(count) {
    ratingCount.textContent = `Ratings: ${count} (local only)`;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function showSuccess(message) {
    footerMsg.textContent = '✓ ' + message;
    footerMsg.style.color = '#28a745';
    setTimeout(() => {
        footerMsg.textContent = 'Ready to rate movies!';
        footerMsg.style.color = '#6c757d';
    }, 2000);
}

function showError(message) {
    footerMsg.textContent = '✗ ' + message;
    footerMsg.style.color = '#dc3545';
    setTimeout(() => {
        footerMsg.textContent = 'Ready to rate movies!';
        footerMsg.style.color = '#6c757d';
    }, 3000);
}


// Recommendations functionality
async function fetchRecommendations() {
    try {
        // Check if there are rated movies
        if (Object.keys(state.ratedMovies).length === 0) {
            showError('Please rate some movies first!');
            return;
        }
        
        // Disable button and show loading
        recommendBtn.disabled = true;
        recommendBtn.textContent = '⏳ Getting recommendations...';
        
        // Open modal and show loading
        recommendationsModal.classList.add('show');
        recommendationsContainer.innerHTML = '<div class="loading">🔄 Training model and generating recommendations...<br><small>This may take a moment</small></div>';
        
        // Convert ratedMovies object to array of rating objects
        const ratings = Object.entries(state.ratedMovies).map(([movieId, rating]) => ({
            user_id: state.userId,
            movie_id: parseInt(movieId),
            rating: rating
        }));
        
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: state.userId,
                top_k: 20,
                ratings: ratings
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get recommendations');
        }
        
        displayRecommendations(data.recommendations);
        
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        recommendationsContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        recommendBtn.disabled = false;
        recommendBtn.textContent = '✨ Get Recommendations';
    }
}

function displayRecommendations(recommendations) {
    if (recommendations.length === 0) {
        recommendationsContainer.innerHTML = '<div class="error-message">No recommendations available.</div>';
        return;
    }
    
    recommendationsContainer.innerHTML = recommendations.map((rec, index) => `
        <div class="recommendation-item">
            <div class="recommendation-info">
                <div class="recommendation-title">${index + 1}. ${escapeHtml(rec.title)}</div>
                <div class="recommendation-score">Recommendation Score: ${(rec.score * 100).toFixed(1)}%</div>
            </div>
            <div class="recommendation-badge">${(rec.score * 100).toFixed(0)}%</div>
        </div>
    `).join('');
}

function closeRecommendations() {
    recommendationsModal.classList.remove('show');
}

// Close modal when clicking outside it
recommendationsModal.addEventListener('click', (e) => {
    if (e.target === recommendationsModal) {
        closeRecommendations();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && recommendationsModal.classList.contains('show')) {
        closeRecommendations();
    }
});
