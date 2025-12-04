import streamlit as st

class MovieSelector3000UI:
    def __init__(self):
        st.set_page_config(page_title="Movie Tinder", page_icon="🎬", layout="centered")
        st.title("Movie Selector 3000")
        st.write("Discover movies tailored to your taste!")

    def render_rating_screen(self, movies):
        """
        Render the movie rating screen where users can dislike, like,
        or superlike the movies displayed in a searchable table.
        :param movies: A list of movie ids ziped with titles to display.
        """
        st.subheader("Rate Movies")
        st.write("Please rate the following movies:")
        
        for movie_id, title in movies:
            st.write(f"**{title}** (ID: {movie_id})")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(f"👎 Dislike {movie_id}", use_container_width=True):
                    st.session_state.ratings[movie_id] = "dislike"
                    st.success(f"You disliked {title}.")

            with col2:
                if st.button(f"👍 Like {movie_id}", use_container_width=True):
                    st.session_state.ratings[movie_id] = "like"
                    st.success(f"You liked {title}.")

            with col3:
                if st.button(f"🌟 Superlike {movie_id}", use_container_width=True):
                    st.session_state.ratings[movie_id] = "superlike"
                    st.success(f"You superliked {title}.")


movieUI = MovieSelector3000UI()