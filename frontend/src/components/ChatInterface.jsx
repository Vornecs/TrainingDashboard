import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Bot, User, AlertCircle } from 'lucide-react';
import './ChatInterface.css';

function ChatInterface({ apiUrl }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxLength, setMaxLength] = useState(100);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userMessage,
          max_length: maxLength,
          temperature: temperature
        })
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.generated
        }]);
      } else {
        const error = await response.json();
        setMessages(prev => [...prev, {
          role: 'error',
          content: `Error: ${error.detail}`
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'error',
        content: `Error: ${error.message}`
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-interface fade-in">
      <div className="section-header">
        <h2>
          <MessageSquare size={24} />
          Chat with Your AI
        </h2>
        <p>Test your trained model in real-time</p>
      </div>

      {/* Settings */}
      <div className="chat-settings">
        <div className="setting-group">
          <label>
            Temperature: {temperature.toFixed(2)}
            <span className="setting-hint">Higher = more creative</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </div>
        <div className="setting-group">
          <label>
            Max Length: {maxLength}
            <span className="setting-hint">Maximum tokens to generate</span>
          </label>
          <input
            type="range"
            min="-1"
            max="1000"
            step="10"
            value={maxLength}
            onChange={(e) => setMaxLength(parseInt(e.target.value))}
          />
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="no-messages">
            <Bot size={48} />
            <h3>Start a Conversation</h3>
            <p>Type a message to chat with your trained AI model</p>
            <div className="example-prompts">
              <button onClick={() => setInput('Hello! How are you?')} className="example-prompt">
                Hello! How are you?
              </button>
              <button onClick={() => setInput('Tell me about transformers')} className="example-prompt">
                Tell me about transformers
              </button>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, idx) => (
              <div key={idx} className={`message message-${message.role}`}>
                <div className="message-icon">
                  {message.role === 'user' && <User size={20} />}
                  {message.role === 'assistant' && <Bot size={20} />}
                  {message.role === 'error' && <AlertCircle size={20} />}
                </div>
                <div className="message-content">
                  <div className="message-role">
                    {message.role === 'user' && 'You'}
                    {message.role === 'assistant' && 'AI'}
                    {message.role === 'error' && 'Error'}
                  </div>
                  <div className="message-text">{message.content}</div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="message message-assistant loading">
                <div className="message-icon">
                  <Bot size={20} />
                </div>
                <div className="message-content">
                  <div className="message-role">AI</div>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input */}
      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message here... (Shift+Enter for new line)"
          rows={3}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || loading}
          className="btn-primary send-button"
        >
          {loading ? <div className="spinner" /> : <Send size={20} />}
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatInterface;
