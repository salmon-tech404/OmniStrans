import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
print("=== ALL WASAPI DEVICES ===")
try:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    print(f"Default WASAPI Output Index: {wasapi_info['defaultOutputDevice']}")
    print(f"Default WASAPI Input Index: {wasapi_info['defaultInputDevice']}")
except Exception as e:
    print(f"Error getting WASAPI info: {e}")

print("\n=== SYSTEM DEFAULT DEVICES (MME) ===")
try:
    print(f"Default Output: {p.get_default_output_device_info()['name']}")
    print(f"Default Input: {p.get_default_input_device_info()['name']}")
except Exception as e:
    print(f"Error getting MME default info: {e}")

print("\n=== DETECTED DEVICES ===")
for i in range(p.get_device_count()):
    try:
        info = p.get_device_info_by_index(i)
        host_api_name = p.get_host_api_info_by_index(info["hostApi"])["name"]
        print(f"Index {i}: {info['name']} | HostAPI: {host_api_name} | MaxIn: {info['maxInputChannels']} | MaxOut: {info['maxOutputChannels']} | Loopback: {info.get('isLoopbackDevice', False)}")
    except Exception as e:
        print(f"Error getting device {i}: {e}")

print("\n=== WASAPI LOOPBACK DEVICES ===")
for loopback in p.get_loopback_device_info_generator():
    print(f"Index {loopback['index']}: {loopback['name']} | Channels: {loopback['maxInputChannels']}")

p.terminate()
