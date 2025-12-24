import "../app.css";
import { useState } from "react";
import IsaacStreamViewer from "../components/IsaacStreamViewer";
import ChatBox from "../components/ChatBox.jsx";

export default function SimulationPage() {
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [connectionError, setConnectionError] = useState(null);

  const sidebarWidth = "20rem"; 

  const handleStatusChange = (status, errMsg = null) => {
    setConnectionStatus(status);
    setConnectionError(errMsg);
  };
  
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* MAIN BODY */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* STATIC SIDEBAR */}
        <aside
          style={{
            width: sidebarWidth,
            height: '100%',
            background: 'var(--bg-secondary)',
            borderRight: '1px solid var(--border-subtle)',
            display: 'flex',
            flexDirection: 'column',
            flexShrink: 0,
          }}
        >
          <ChatBox />
        </aside>

        {/* VIDEO STREAM */}
        <main 
          style={{ 
            flex: 1,
            position: 'relative',
            background: 'var(--bg-primary)',
          }}
        >
          <IsaacStreamViewer 
            onStatusChange={handleStatusChange}
            connectionStatus={connectionStatus}
          />
          
          {/* Connection status badge */}
          <div 
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 12px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              zIndex: 10,
            }}
          >
            <div className={`status-dot ${connectionStatus}`}></div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              {connectionStatus === "connected" && "Live"}
              {connectionStatus === "connecting" && "Connecting..."}
              {connectionStatus === "disconnected" && "Disconnected"}
              {connectionStatus === "error" && "Error"}
            </span>
          </div>
        </main>
      </div>
    </div>
  );
}