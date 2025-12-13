# save this as audio_downloader.py
import yt_dlp
import sys
import os


def download_youtube_audio(url, output_dir="downloads"):
    """
    Download audio from a YouTube URL and convert to MP3.

    Args:
        url (str): YouTube video or playlist URL
        output_dir (str): Directory to save downloaded files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from YouTube: {url}")
            ydl.download([url])
        print(f"YouTube download completed! Files saved in '{output_dir}' folder")
    except Exception as e:
        print(f"An error occurred during YouTube download: {e}")
        return False
    return True


def download_soundcloud_mp3(url, output_dir="downloads"):
    """
    Download audio from a SoundCloud URL and convert to MP3.

    Args:
        url (str): SoundCloud track or playlist URL
        output_dir (str): Directory to save downloaded files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading from SoundCloud: {url}")
            ydl.download([url])
        print(f"SoundCloud download completed! Files saved in '{output_dir}' folder")
    except Exception as e:
        print(f"An error occurred during SoundCloud download: {e}")
        return False
    return True


def download_audio(url, platform=None, output_dir="downloads"):
    """
    Universal function to download audio from YouTube or SoundCloud.
    Automatically detects the platform or uses specified platform.

    Args:
        url (str): URL of the audio content
        platform (str, optional): 'youtube' or 'soundcloud'. If None, auto-detects.
        output_dir (str): Directory to save downloaded files
    """
    # Auto-detect platform if not specified
    if platform is None:
        if "youtube.com" in url or "youtu.be" in url:
            platform = "youtube"
        elif "soundcloud.com" in url:
            platform = "soundcloud"
        else:
            print(
                "Error: Could not determine platform from URL. Please specify 'youtube' or 'soundcloud'."
            )
            return False

    # Call appropriate download function
    if platform.lower() == "youtube":
        return download_youtube_audio(url, output_dir)
    elif platform.lower() == "soundcloud":
        return download_soundcloud_mp3(url, output_dir)
    else:
        print(
            f"Error: Unsupported platform '{platform}'. Use 'youtube' or 'soundcloud'."
        )
        return False


def main():
    """Main function with command line interface."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python audio_downloader.py <url> [platform] [output_dir]")
        print("")
        print("Examples:")
        print("  python audio_downloader.py https://www.youtube.com/watch?v=example")
        print(
            "  python audio_downloader.py https://soundcloud.com/artist/track soundcloud"
        )
        print("  python audio_downloader.py https://youtu.be/example youtube my_music")
        print("")
        print("Platform is optional (auto-detected from URL).")
        sys.exit(1)

    # Parse arguments
    url = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "downloads"

    # Download the audio
    success = download_audio(url, platform, output_dir)

    if success:
        print("\nAudio download completed successfully!")
    else:
        print("\nAudio download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
