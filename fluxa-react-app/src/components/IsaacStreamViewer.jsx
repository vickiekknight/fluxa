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
          // Server address (without port)
          server: "10.10.10.4",
          
          // Port for WebRTC signaling
          port: 49100,
          
          // Attach to video element
          videoElement: videoRef.current,
          
          // Display settings
          width: 1920,
          height: 1080,
          fps: 60,

          onStart: (msg) => {
            console.log("✅ Stream started:", msg);
            if (mounted) onStatusChange("connected");
          },
          
          onStop: (msg) => {
            console.log("Stream stopped:", msg);
            if (mounted) onStatusChange("disconnected");
          },
          
          onError: (msg) => {
            console.error("Stream error:", msg);
            const errMsg = msg?.message || "Stream error occurred";
            setError(errMsg);
            if (mounted) onStatusChange("error", errMsg);
          },
          
          onUpdate: (msg) => {
            // Stream is actively sending frames
            // You can add frame rate monitoring here if needed
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

    // Small delay to ensure video element is mounted
    const timer = setTimeout(() => {
      startStream();
    }, 100);

    // Cleanup
    return () => {
      mounted = false;
      clearTimeout(timer);
      
      // Disconnect stream
      if (streamerRef.current) {
        try {
          streamerRef.current.stop();
          console.log("Stream disconnected");
        } catch (err) {
          console.error("Error disconnecting stream:", err);
        }
      }
    };
  }, [onStatusChange, connectionStatus]);

  return (
    <div className="h-full flex flex-col">
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 max-w-md">
          <div className="bg-red-900/90 border border-red-500 text-red-100 rounded-lg p-4 shadow-lg">
            <p className="font-bold mb-1">Connection Error</p>
            <p className="text-sm">{error}</p>
            <p className="text-xs mt-2 opacity-80">
              Check that Isaac Sim is running with livestream enabled on 10.10.10.4:49100
            </p>
          </div>
        </div>
      )}
      <div className="flex-1 relative bg-black">
        <video
          ref={videoRef}
          id="remote-video"
          className="absolute inset-0 w-full h-full object-contain"
          playsInline
          muted
          autoPlay
        />
        
        {/* Loading overlay */}
        {connectionStatus === "connecting" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <div className="text-center">
              <div className="spinner mb-4"></div>
              <p className="text-green-400">Connecting to Isaac Sim...</p>
              <p className="text-green-400/60 text-sm mt-2">10.10.10.4:49100</p>
            </div>
          </div>
        )}
        
        {/* Disconnected overlay */}
        {connectionStatus === "disconnected" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <div className="text-center">
              <p className="text-yellow-400 text-lg">Stream Disconnected</p>
              <p className="text-yellow-400/60 text-sm mt-2">Refresh to reconnect</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}