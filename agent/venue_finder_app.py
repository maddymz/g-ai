import streamlit as st
import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add agent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from venue_finder_agent import run_venue_search

# Page config
st.set_page_config(
    page_title="Conference Venue Finder",
    page_icon="🏢",
    layout="wide"
)

# Title and description
st.title("🏢 Conference Venue Finder")
st.markdown("Find the perfect venue using AI agents powered by CrewAI")
st.markdown("---")

# Input section
with st.form("venue_search_form"):
    conference_name = st.text_input(
        "Conference Name",
        placeholder="e.g., AI Innovations Summit",
        help="Enter the name of your conference"
    )

    requirements = st.text_area(
        "Requirements",
        placeholder="e.g., Capacity for 500 people, downtown location, A/V equipment, parking, budget $20,000",
        height=150,
        help="Describe your venue requirements in detail (capacity, location, amenities, budget, etc.)"
    )

    submitted = st.form_submit_button("Search for Venues", type="primary")

# Execution section
if submitted:
    if conference_name and requirements:
        # Capture agent logs
        log_capture = StringIO()

        with st.spinner('🔍 AI agents are searching and analyzing venues...'):
            try:
                # Redirect stdout/stderr to capture verbose logs
                with redirect_stdout(log_capture), redirect_stderr(log_capture):
                    result = run_venue_search(conference_name, requirements)

                # Display results
                st.success("✅ Venue search completed!")
                st.markdown("### Recommended Venues")
                st.markdown(result)

                # Show agent logs in expander
                with st.expander("🔍 View Agent Logs and Reasoning"):
                    logs = log_capture.getvalue()
                    if logs:
                        st.text(logs)
                    else:
                        st.info("No detailed logs available")

            except Exception as e:
                st.error(f"❌ Error during venue search: {str(e)}")
                st.exception(e)

                # Show any captured logs even on error
                logs = log_capture.getvalue()
                if logs:
                    with st.expander("🔍 View Error Logs"):
                        st.text(logs)
    else:
        st.warning("⚠️ Please fill in both conference name and requirements")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses two AI agents working together:

    **1. Venue Finder Agent**
    - Searches for suitable venues
    - Considers capacity, location, amenities, and pricing
    - Uses web search to find real venue information

    **2. QA Specialist Agent**
    - Reviews and validates venue options
    - Ensures quality standards are met
    - Provides detailed suitability analysis

    Both agents collaborate to deliver the best recommendations for your conference.
    """)

    st.markdown("---")

    st.header("Tips for Best Results")
    st.markdown("""
    Include these details in your requirements:

    - **Capacity**: Expected number of attendees
    - **Location**: City, region, or specific area preferences
    - **Amenities**: A/V equipment, Wi-Fi, catering, breakout rooms
    - **Budget**: Price range or maximum budget
    - **Dates**: Preferred dates or time of year
    - **Accessibility**: Parking, public transport, accessibility features
    - **Style**: Conference hall, hotel, outdoor venue, etc.
    """)

    st.markdown("---")

    st.header("Technical Details")
    st.markdown("""
    **Powered by:**
    - CrewAI multi-agent framework
    - Tavily web search API
    - OpenAI language models

    **Processing time:** 30-60 seconds

    Search results are gathered in real-time from web sources.
    """)

# Footer
st.markdown("---")
st.markdown("🤖 Built with CrewAI and Streamlit | [View Source](https://github.com)")
