# VHS tape for assets/demo.gif — https://github.com/charmbracelet/vhs
#
# Regenerate after UI changes with:
#
#     vhs assets/demo.tape
#
# Requirements: vhs on PATH, tts-daemon installed (the demo uses the
# dependency-free tone provider, so no TTS engine is needed).

Output assets/demo.gif

Set Theme "Catppuccin Mocha"
Set FontSize 20
Set Width 800
Set Height 480
Set Padding 16
Set TypingSpeed 40ms

# Start the server quietly in the background, then present a clean prompt.
Hide
Type "tts-daemon serve > /tmp/tts-daemon-demo.log 2>&1 &"
Enter
Sleep 2s
Type "clear"
Enter
Show

Type `tts-daemon speak "Hello! I am your local text to speech gateway."`
Enter
Sleep 2.5s

Type `tts-daemon speak --interrupt "Stop everything — this one is more important."`
Enter
Sleep 2.5s

Type "tts-daemon status"
Enter
Sleep 3s

Type `curl -X POST localhost:5111/v1/speak -H 'content-type: application/json' -d '{"text": "Anything can talk to it."}'`
Enter
Sleep 4s

Hide
Type "kill %1"
Enter
