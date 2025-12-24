import { useEffect, useState, useRef } from "react";
import { AppStreamer, StreamType, LogLevel } from "@nvidia/omniverse-webrtc-streaming-library";

export default function IsaacStreamViewer({ onStatusChange, connectionStatus }) {
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const streamerRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    async function startStream() {
      if (!videoRef.current) {
        console.error("Video element not ready");
        return;
      }
      onStatusChange("connecting");

      const streamParams = {
        streamSource: StreamType.DIRECT,
        logLevel: LogLevel.WARN,
        streamConfig: {
          server: "10.10.10.5",
          videoElement: videoRef.current,
          width: 1920,
          height: 1080,
          fps: 60,

          onStart: (msg) => {
            console.log("✅ Stream started:", msg);
            
            if (msg.status === 'warning') {
              console.warn("⚠️ Stream started with warning - connection may be unstable");
              setError("Connection unstable - retrying...");
              return;
            }
            
            if (msg.status === 'success' || msg.action === 'start') {
              if (videoRef.current) {
                console.log("📹 Video srcObject:", videoRef.current.srcObject);
                console.log("📹 Video readyState:", videoRef.current.readyState);
              }
              if (mounted) onStatusChange("connected");
            }
          },
          
          onVideo: (msg) => {
            console.log("🎥 Video stream received:", msg);
            if (mounted && connectionStatus !== "connected") {
              onStatusChange("connected");
              setError(null);
            }
          },
          
          onUpdate: (msg) => {
            console.log("📊 Stream update:", msg);
          },
          
          onStop: (msg) => {
            console.log("🛑 Stream stopped:", msg);
            if (mounted) onStatusChange("disconnected");
          },
          
          onError: (msg) => {
            console.error("❌ Stream error:", msg);
            const errMsg = msg?.message || msg?.info || "Stream error occurred";
            setError(errMsg);
            if (mounted) onStatusChange("error", errMsg);
          },
        },
      };

      try {
        console.log("Connecting to Isaac Sim stream...");
        const streamer = await AppStreamer.connect(streamParams);
        streamerRef.current = streamer;
        console.log("Stream connection initiated");
      } catch (err) {
        console.error("Failed to connect:", err);
        const errMsg = err.message || "Failed to connect to stream";
        setError(errMsg);
        if (mounted) onStatusChange("error", errMsg);
      }
    }

    const timer = setTimeout(() => {
      startStream();
    }, 100);

    return () => {
      mounted = false;
      clearTimeout(timer);
      
      if (streamerRef.current) {
        streamerRef.current = null;
        console.log("Stream reference cleared");
      }
    };
  }, [onStatusChange]);

  return (
    <div className="h-full flex flex-col bg-primary">
      <div className="flex-1 relative">
        <video
          ref={videoRef}
          id="remote-video"
          className="absolute inset-0 w-full h-full"
          style={{ objectFit: 'contain', background: '#0a0a0b' }}
          playsInline
          muted
          autoPlay
        />
        
        {/* Connecting overlay */}
        {connectionStatus === "connecting" && (
          <div className="overlay">
            <div className="overlay-content">
              <div className="spinner mb-4"></div>
              <p className="overlay-title">Connecting to Isaac Sim</p>
              <p className="overlay-subtitle">Establishing WebRTC connection...</p>
            </div>
          </div>
        )}
        
        {/* Disconnected overlay */}
        {connectionStatus === "disconnected" && (
          <div className="overlay">
            <div className="overlay-content">
              <svg 
                width="48" 
                height="48" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="var(--text-muted)" 
                strokeWidth="1.5"
                style={{ marginBottom: '16px' }}
              >
                <path d="M18.36 6.64a9 9 0 1 1-12.73 0M12 2v10"/>
              </svg>
              <p className="overlay-title">Stream Disconnected</p>
              <p className="overlay-subtitle">Refresh the page to reconnect</p>
              <button 
                className="btn-primary mt-2"
                onClick={() => window.location.reload()}
                style={{ marginTop: '16px' }}
              >
                Reconnect
              </button>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {connectionStatus === "error" && (
          <div className="overlay">
            <div className="overlay-content">
              <svg 
                width="48" 
                height="48" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="var(--error)" 
                strokeWidth="1.5"
                style={{ marginBottom: '16px' }}
              >
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4M12 16h.01"/>
              </svg>
              <p className="overlay-title">Connection Error</p>
              <p className="overlay-subtitle">{error || "Failed to connect to stream"}</p>
              <button 
                className="btn-primary"
                onClick={() => window.location.reload()}
                style={{ marginTop: '16px' }}
              >
                Try Again
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}