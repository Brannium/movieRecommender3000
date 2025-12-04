import streamlit as st
import math

class MovieSelector3000UI:
    ITEMS_PER_PAGE = 20

    def __init__(self):
        st.set_page_config(page_title="Movie Tinder", page_icon="🎬", layout="centered")
        st.title("Movie Selector 3000")
        st.write("Discover movies tailored to your taste!")
        
        # Custom CSS for button styling
        st.markdown("""
        <style>
        .disliked { background-color: #ffcccc !important; }
        .liked { background-color: #ccffcc !important; }
        </style>
        """, unsafe_allow_html=True)

    def render_rating_screen(self, movies):
        """
        Render the movie rating screen with pagination (20 items per page).
        Users can navigate between pages and rate movies using dislike, like, or superlike.
        :param movies: A list of movie ids zipped with titles to display.
        """
        st.subheader("Rate Movies")
        
        # Initialize pagination state
        if "current_page" not in st.session_state:
            st.session_state.current_page = 0
        
        total_movies = len(movies)
        total_pages = math.ceil(total_movies / self.ITEMS_PER_PAGE)
        
        if total_pages == 0:
            st.write("No movies to rate.")
            return
        
        # Display page info and pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Previous", use_container_width=True):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
                    st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        
        with col3:
            if st.button("Next →", use_container_width=True):
                if st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page += 1
                    st.rerun()
        
        # Calculate which movies to display on current page
        start_idx = st.session_state.current_page * self.ITEMS_PER_PAGE
        end_idx = min(start_idx + self.ITEMS_PER_PAGE, total_movies)
        page_movies = movies[start_idx:end_idx]
        
        st.write(f"Showing {len(page_movies)} of {total_movies} movies")
        st.divider()
        
        # Render only the movies for the current page
        for movie_id, title in page_movies:
            col1, col2, col3, col4 = st.columns(4)
            
            current_rating = st.session_state.ratings.get(movie_id, 0)

            with col1:
                st.write(f"**{title}**")
            with col2:
                if current_rating == -1:
                    st.markdown('<div style="background-color: #ffcccc; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">👎 Dislike</div>', unsafe_allow_html=True)
                if st.button(f"👎 Dislike", use_container_width=True, key=f"dislike_{movie_id}"):
                    st.session_state.ratings[movie_id] = 0 if current_rating == -1 else -1
                    st.rerun()

            with col3:
                if current_rating == 1:
                    st.markdown('<div style="background-color: #ccffcc; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">👍 Like</div>', unsafe_allow_html=True)
                if st.button(f"👍 Like", use_container_width=True, key=f"like_{movie_id}"):
                    st.session_state.ratings[movie_id] = 0 if current_rating == 1 else 1
                    st.rerun()

            with col4:
                if current_rating == 2:
                    st.markdown('<div style="background-color: #ccffcc; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">🌟 Superlike</div>', unsafe_allow_html=True)
                if st.button(f"🌟 Superlike", use_container_width=True, key=f"superlike_{movie_id}"):
                    st.session_state.ratings[movie_id] = 0 if current_rating == 2 else 2
                    st.rerun()
