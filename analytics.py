import streamlit.components.v1 as components


def track_visit(page):
    """Count a page view in GoatCounter (dashboard: https://fisherlab.goatcounter.com)."""
    components.html(
        f'<img src="https://fisherlab.goatcounter.com/count?p=/emotracknlp/{page}&t=EmoTrackNLP%20-%20{page}" '
        'width="1" height="1" alt="">',
        height=0,
    )
