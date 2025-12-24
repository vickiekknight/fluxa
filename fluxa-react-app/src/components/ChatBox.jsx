import { useState, useEffect, useRef } from "react";

export default function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://10.10.10.5:4373/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();

      if (data.error) throw new Error(data.error);

      setMessages((prev) => [
        ...prev,
        { sender: "agent", text: data.message },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "error", text: `Error: ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */} 
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        flexShrink: 0,
      }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Fluxa
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
          robotics synthetic data with natural language
        </div>
      </div>

      {/* Messages - scrollable area */}
      <div style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        padding: '16px',
      }}>
        {messages.length === 0 ? (
          <div style={{ padding: '24px 0', textAlign: 'center' }}>
            <p style={{ 
              fontSize: '0.875rem', 
              fontWeight: 500, 
              color: 'var(--text-secondary)',
              marginBottom: '16px',
            }}>
              What will you create today?
            </p>
            <p style={{ 
              fontSize: '0.75rem', 
              color: 'var(--text-muted)',
              marginBottom: '8px',
            }}>
              Example commands:
            </p>
            <ul style={{
              textAlign: 'left',
              fontSize: '0.75rem',
              color: 'var(--text-tertiary)',
              fontFamily: "'SF Mono', 'Fira Code', monospace",
              listStyle: 'none',
              padding: 0,
              margin: 0,
            }}>
              <li style={{ padding: '4px 0' }}>
                <span style={{ color: 'var(--accent)', marginRight: '8px' }}>→</span>
                Create a Franka robot at [0, 0, 0]
              </li>
              <li style={{ padding: '4px 0' }}>
                <span style={{ color: 'var(--accent)', marginRight: '8px' }}>→</span>
                Add a physics scene with floor
              </li>
              <li style={{ padding: '4px 0' }}>
                <span style={{ color: 'var(--accent)', marginRight: '8px' }}>→</span>
                Create 3 robots in a row
              </li>
              <li style={{ padding: '4px 0' }}>
                <span style={{ color: 'var(--accent)', marginRight: '8px' }}>→</span>
                Add better lighting
              </li>
            </ul>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  fontSize: '0.875rem',
                  lineHeight: 1.5,
                  maxWidth: '85%',
                  ...(msg.sender === "user" ? {
                    background: 'var(--accent-muted)',
                    color: 'var(--text-primary)',
                    marginLeft: 'auto',
                    borderBottomRightRadius: '4px',
                  } : msg.sender === "error" ? {
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    color: '#fca5a5',
                  } : {
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-subtle)',
                    borderBottomLeftRadius: '4px',
                  })
                }}
              >
                {msg.text}
              </div>
            ))}
            {loading && (
              <div style={{
                padding: '10px 14px',
                borderRadius: '8px',
                background: 'var(--bg-tertiary)',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <div className="spinner-sm"></div>
                <span>Processing...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input - fixed at bottom */}
      <form
        onSubmit={handleSend}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px',
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-secondary)',
          flexShrink: 0,
        }}
      >
        <input
          type="text"
          style={{
            flex: 1,
            padding: '10px 12px',
            fontSize: '0.875rem',
            color: 'var(--text-primary)',
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
            outline: 'none',
          }}
          placeholder="Talk to Fluxa"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '10px 16px',
            fontSize: '0.8125rem',
            fontWeight: 500,
            color: '#0a0a0b',
            background: 'var(--accent)',
            border: '1px solid var(--accent)',
            borderRadius: '8px',
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            opacity: loading || !input.trim() ? 0.5 : 1,
          }}
        >
          {loading ? (
            <div className="spinner-sm" style={{ borderTopColor: '#0a0a0b' }}></div>
          ) : (
            <>
              Send
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
              </svg>
            </>
          )}
        </button>
      </form>
    </div>
  );
}