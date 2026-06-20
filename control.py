import argparse
import os
import requests
import json

def get_auth_header():
    token = os.getenv("TTS_AUTHORIZATION_TOKEN", "vote_goof_2024")
    return {"Authorization": token}

def get_voice_display_name(voice):
    if isinstance(voice, dict):
        return voice.get("name") or voice.get("voice") or voice.get("latent") or str(voice)
    return str(voice)

def get_voice_request_value(voice):
    if isinstance(voice, dict):
        return voice.get("name") or voice.get("voice") or voice.get("latent") or ""
    return str(voice)

def toggle_logging(args):
    url = f"http://{args.host}:{args.port}/toggle-logging"
    params = {}
    if args.level:
        params["level"] = args.level
    try:
        print(f"Requesting logging level change at {url}...")
        response = requests.get(url, headers=get_auth_header(), params=params)
        if response.status_code == 200:
            print("Successfully updated logging levels:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error updating logging: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to API: {e}")

def generate_audio(args):
    base_url = f"http://{args.host}:{args.port}"
    
    # 1. Fetch voices
    try:
        print("Fetching available voices...")
        v_resp = requests.get(f"{base_url}/tts-voices", headers=get_auth_header())
        if v_resp.status_code != 200:
            print(f"Failed to get voices: {v_resp.status_code}")
            return
        voices = v_resp.json()
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return

    # 2. Ask user for voice
    print("\nAvailable voices:")
    for i, v in enumerate(voices):
        print(f"{i+1}. {get_voice_display_name(v)}")
    
    try:
        choice_input = input(f"\nSelect a voice (1-{len(voices)}): ")
        choice = int(choice_input) - 1
        voice = voices[choice]
        voice_name = get_voice_display_name(voice)
        voice_param = get_voice_request_value(voice)
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    if not voice_param:
        print("Selected voice does not contain a usable voice name.")
        return

    # 3. Ask for text
    text = input("Enter text to generate: ")
    if not text:
        print("Text cannot be empty.")
        return

    # 4. Preparation
    identifier = f"cli_{os.getpid()}"
    filters = args.filters
    if args.radio:
        if "radio" not in filters:
            filters = f"{filters}|radio" if filters else "radio"

    params = {
        "voice": voice_param,
        "pitch": args.pitch,
        "special_filters": filters,
        "identifier": identifier
    }

    # 5. Call TTS endpoint
    print(f"Generating base audio for '{voice_name}' (ID: {identifier})...")
    try:
        response = requests.get(
            f"{base_url}/tts",
            headers=get_auth_header(),
            params=params,
            json={"text": text},
            stream=True
        )

        if response.status_code == 200:
            filename = f"output_{identifier}.ogg"
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Success! Base audio saved to {filename}")
        else:
            print(f"Error generating base audio: {response.status_code}")
            print(response.text)
            return

        # 6. If radio requested, fetch from radio endpoint
        if args.radio:
            print(f"Fetching radio processed version for ID: {identifier}...")
            # Note: Radio endpoint just needs identifier in query params
            radio_params = {"identifier": identifier}
            r_response = requests.get(
                f"{base_url}/tts-radio",
                headers=get_auth_header(),
                params=radio_params,
                json={"raw_text": text, "gibberish_text": ""},
                stream=True
            )

            if r_response.status_code == 200:
                radio_filename = f"output_radio_{identifier}.ogg"
                with open(radio_filename, 'wb') as f:
                    for chunk in r_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Success! Radio version saved to {radio_filename}")
            else:
                print(f"Error fetching radio version: {r_response.status_code}")
                print(r_response.text)

    except Exception as e:
        print(f"Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="TTS API Control Script")
    parser.add_argument("--host", default="localhost", help="API Host (default: localhost)")
    parser.add_argument("--port", default="5002", help="API Port (default: 5002)")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Toggle Logging Command
    log_parser = subparsers.add_parser("toggle-logging", help="Toggle or set logging level")
    log_parser.add_argument("level", nargs="?", help="Optional logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # Generate Audio Command
    gen_parser = subparsers.add_parser("generate", help="Generate audio from text")
    gen_parser.add_argument("--pitch", default="0", help="Pitch shift steps (default: 0)")
    gen_parser.add_argument("--filters", default="", help="Special filters (e.g. 'silicon', 'radio')")
    gen_parser.add_argument("--radio", action="store_true", help="Use radio effect endpoint")

    args = parser.parse_args()

    if args.command == "toggle-logging":
        toggle_logging(args)
    elif args.command == "generate":
        generate_audio(args)

if __name__ == "__main__":
    main()
