import { useEffect, useState } from 'react'

export default function App() {
  const [activePage, setActivePage] = useState('Home')
  const isAiChat = activePage === 'AI Chat'
  const isAgentTrace = activePage === 'Agent Trace'
  const [chatInput, setChatInput] = useState('')
  const [lastUserMessage, setLastUserMessage] = useState('')
  const [showLogs, setShowLogs] = useState(false)
  const [agentAnswer, setAgentAnswer] = useState('')
  const [results, setResults] = useState([])
  const [traceSteps, setTraceSteps] = useState([])
  const [showTraceDetails, setShowTraceDetails] = useState(false)
  const [isTraceAnimating, setIsTraceAnimating] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [logs, setLogs] = useState(() => {
    if (typeof window === 'undefined') {
      return []
    }

    const stored = localStorage.getItem('aiChatLogs')
    return stored ? JSON.parse(stored) : []
  })

  useEffect(() => {
    localStorage.setItem('aiChatLogs', JSON.stringify(logs))
  }, [logs])

  const addLog = (type, message) => {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      time: new Date().toLocaleString(),
      type,
      message,
    }

    setLogs((prev) => [entry, ...prev])
  }

  const formatDuration = (durationMs) => {
    if (durationMs === undefined || durationMs === null) {
      return '-'
    }

    if (durationMs < 1000) {
      return `${durationMs} ms`
    }

    return `${(durationMs / 1000).toFixed(2)} s`
  }

  const formatTraceBreakdown = (breakdown = {}) => {
    const labels = {
      planning_ms: 'plan',
      tool_ms: 'tool',
      observe_ms: 'observe',
      generation_ms: 'generate',
    }

    return Object.entries(breakdown)
      .filter(([, value]) => value !== undefined && value !== null)
      .map(([key, value]) => `${labels[key] || key}: ${formatDuration(value)}`)
      .join(' · ')
  }

  const handleChatInputChange = (event) => {
    setChatInput(event.target.value)
  }

  const ragEndpoint = import.meta.env.VITE_RAG_ENDPOINT

  const animateTraceSteps = (steps) => {
    setTraceSteps([])
    setIsTraceAnimating(true)

    steps.forEach((step, index) => {
      setTimeout(() => {
        setTraceSteps((prev) => [...prev, step])

        if (index === steps.length - 1) {
          setIsTraceAnimating(false)
        }
      }, index * 700)
    })
  }

  const handleChatSubmit = async () => {
    const trimmed = chatInput.trim()
    setLastUserMessage(trimmed)
    if (!trimmed) {
      return
    }

    setErrorMessage('')
    setIsLoading(true)
    addLog('input', `User submitted: ${trimmed}`)
    // addLog('processing', `Processing request: ${trimmed}`)
    try {
      let nextResults = []

      if (ragEndpoint) {
        const response = await fetch(ragEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: trimmed, topK: 3 }),
        })

        if (!response.ok) {
          throw new Error(`RAG request failed (${response.status})`)
        }

        const data = await response.json()
        
        setAgentAnswer(data.answer || '')
        animateTraceSteps(data.trace || [])

        if (data.logs) {
          const backendLogs = data.logs.map((log, index) => ({
            id: `${Date.now()}-${index}`,
            time: new Date().toLocaleString(),
            type: log.type || 'agent',
            message: log.message || '',
          }))

          setLogs((prev) => [...backendLogs.reverse(), ...prev])
        }
        nextResults = (data.results || []).map((item) => ({
        title: item.title,
        match: item.match,
        href: item.url || item.href,
        imageUrl: item.image_url || item.imageUrl || '',
        reason: item.reason || '',
      }))
      } else {
        nextResults = [
          {
            title: 'Violet Evergarden OST',
            match: 92,
            href: 'https://www.youtube.com/results?search_query=Violet+Evergarden+OST',
          },
          {
            title: 'Angel Beats! OST',
            match: 92,
            href: 'https://www.youtube.com/results?search_query=Angel+Beats+OST',
          },
          {
            title: 'Garden of Words OST',
            match: 92,
            href: 'https://www.youtube.com/results?search_query=Garden+of+Words+OST',
          },
          {
            title: 'Clannad OST',
            match: 92,
            href: 'https://www.youtube.com/results?search_query=Clannad+OST',
          },
        ]
      }

      setResults(nextResults)
      // addLog('result', 'Processing complete: generated OST recommendations.')
      setChatInput('')
    } catch (error) {
      setErrorMessage(error.message || 'RAG request failed')
      addLog('error', error.message || 'RAG request failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#050816] text-white flex">
      <aside className="w-[280px] border-r border-white/10 bg-[#070B1B] p-6">
        <div className="flex items-center gap-3 mb-12 min-w-0">
          <div className="w-12 h-12 rounded-2xl bg-violet-600 flex items-center justify-center text-white font-semibold">
            AI
          </div>
          <div className="min-w-0">
            <h1 className="text-base font-bold leading-tight break-words">
              Anime_OST_Scout_Agent
            </h1>
            <p className="text-xs text-gray-400">
              AI Agent for Soundtrack Discovery
            </p>
          </div>
        </div>

        <div className="space-y-3">
          {['Home', 'AI Chat', 'Explore OST', 'My Library'].map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setActivePage(item)}
              className={`w-full text-left px-5 py-4 rounded-2xl transition-colors ${
                activePage === item
                  ? 'bg-violet-600/20 border border-violet-500/30'
                  : 'hover:bg-white/5 text-gray-300'
              }`}
            >
              {item}
            </button>
          ))}
        </div>

        <div className="mt-10">
          <div className="text-xs uppercase tracking-[0.2em] text-gray-500 mb-3">
            AI System
          </div>
          <button
            type="button"
            onClick={() => setActivePage('Agent Trace')}
            className={`w-full text-left px-5 py-4 rounded-2xl transition-colors ${
              activePage === 'Agent Trace'
                ? 'bg-violet-600/20 border border-violet-500/30'
                : 'hover:bg-white/5 text-gray-300'
            }`}
          >
            Agent Trace
          </button>
        </div>
      </aside>

      <main className="flex-1 p-6">
        {isAiChat ? (
          <div className="rounded-3xl border border-white/10 bg-[#0A1023] p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold">AI Chat</h2>
              <button
                type="button"
                onClick={() => setShowLogs((prev) => !prev)}
                className="px-4 py-2 rounded-2xl text-xs text-gray-300 border border-white/10 hover:bg-white/5"
              >
                {showLogs ? 'ซ่อน Logging' : 'ดู Logging'}
              </button>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
              {agentAnswer ? (
                <div className="mb-6 rounded-2xl border border-violet-500/30 bg-violet-500/10 p-5">
                  <div className="text-sm font-semibold text-violet-200 mb-2">
                    Scout Agent Response
                  </div>
                  <p className="text-sm leading-7 text-gray-200 whitespace-pre-line">
                    {agentAnswer}
                  </p>
                </div>
              ) : null}

              {results.length === 0 ? (
                <p className="text-gray-400">
                  พิมพ์คำถามถึง AI เพื่อเริ่มค้นหา OST ที่ใช่สำหรับคุณ
                </p>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {results.map((item) => (
                    <div
                      key={item.title}
                      className="rounded-2xl overflow-hidden bg-white/5 border border-white/10"
                    >
                      <div className="h-40 bg-gradient-to-br from-violet-500/30 to-fuchsia-500/20 overflow-hidden relative">
                        <div className="absolute inset-0 flex items-center justify-center text-violet-200 text-sm">
                          No Image
                        </div>

                        {item.imageUrl ? (
                          <img
                            src={item.imageUrl}
                            alt={item.title}
                            className="relative h-full w-full object-cover"
                            onError={(event) => {
                              event.currentTarget.style.display = 'none'
                            }}
                          />
                        ) : null}
                      </div>
                      <div className="p-4">
                        <h3>{item.title}</h3>
                        <p className="text-emerald-400 text-sm mt-2">
                          {item.match}% match
                        </p>

                        {item.reason ? (
                          <p className="text-xs text-gray-400 mt-2 line-clamp-3">
                            {item.reason}
                          </p>
                        ) : null}
                        <a
                          href={item.href}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center mt-3 text-sm text-violet-300 hover:text-violet-200"
                        >
                          ฟังตัวอย่าง
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-6 rounded-2xl bg-black/30 border border-violet-500/30 p-2 flex items-center">
                <input
                  value={chatInput}
                  onChange={handleChatInputChange}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      handleChatSubmit()
                    }
                  }}
                  disabled={isLoading}
                  placeholder="พิมพ์คำถามถึง AI..."
                  className="flex-1 bg-transparent px-5 py-4 outline-none"
                />

                <button
                  type="button"
                  onClick={handleChatSubmit}
                  disabled={isLoading}
                  className="w-14 h-14 rounded-xl bg-violet-600 disabled:opacity-60"
                >
                  ➤
                </button>
              </div>

              {isLoading ? (
                <div className="mt-3 text-sm text-gray-400">กำลังประมวลผล...</div>
              ) : null}

              {errorMessage ? (
                <div className="mt-3 text-sm text-rose-300">{errorMessage}</div>
              ) : null}

              {showLogs ? (
                <div className="mt-6 rounded-2xl border border-white/10 bg-[#0B1126] p-4 max-h-64 overflow-auto">
                  <div className="text-sm font-semibold mb-3">Logging</div>
                  {logs.length === 0 ? (
                    <div className="text-sm text-gray-400">ยังไม่มีข้อมูล logging</div>
                  ) : (
                    <div className="space-y-3 text-sm">
                      {logs.map((entry) => (
                        <div key={entry.id} className="rounded-xl border border-white/10 bg-white/5 p-3">
                          <div className="text-xs text-gray-400">{entry.time}</div>
                          <div className="mt-1">
                            <span className="text-violet-300">[{entry.type}]</span> {entry.message}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        ) : isAgentTrace ? (
          <div className="rounded-3xl border border-white/10 bg-[#0A1023] p-6 shadow-[0_0_40px_rgba(88,101,242,0.08)]">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2 text-sm text-violet-200">
                <span className="text-violet-300">✦</span>
                <span className="font-semibold">Agent Trace</span>
                <span className="ml-2 px-2 py-0.5 rounded-full bg-violet-500/20 text-[10px] text-violet-200">
                  LIVE
                </span>
              </div>
              <button type="button" className="text-xs text-gray-400 hover:text-gray-200">
                ⤴
              </button>
            </div>

            <div className="space-y-4">
              {traceSteps.map((step, index) => (
                <div key={`${step.step}-${index}`} className="flex items-center gap-4">
                  <div
                    className={`w-10 h-10 rounded-full text-xs font-semibold flex items-center justify-center text-[#050816]
                    ${
                      step.status === 'success'
                        ? 'bg-gradient-to-br from-emerald-500 to-lime-300'
                        : 'bg-gradient-to-br from-violet-500 to-fuchsia-300'
                    }`}
                  >
                    {index + 1}
                  </div>

                  <div className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 flex items-center justify-between gap-4">
                    <div>
                      <div className="text-sm font-semibold text-gray-100">
                        {step.step}
                      </div>

                      <div className="text-xs text-gray-400 mt-1">
                        {step.reasoning || step.feedback || 'Agent processing'}
                      </div>
                    </div>

                    <div className="flex flex-col items-end gap-1 text-xs text-gray-400">
                      <span>{step.status}</span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-violet-200">
                        {formatDuration(step.duration_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
              {isTraceAnimating ? (
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-300 animate-pulse" />

                  <div className="flex-1 rounded-2xl border border-violet-500/30 bg-violet-500/10 px-4 py-3">
                    <div className="text-sm font-semibold text-violet-200">
                      Agent is thinking...
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      กำลังประมวลผลขั้นตอนถัดไป
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            <button
              type="button"
              onClick={() => setShowTraceDetails((prev) => !prev)}
              className="mt-6 w-full rounded-2xl border border-violet-500/30 bg-violet-500/10 py-3 text-sm text-violet-200 hover:bg-violet-500/20 transition-colors"
            >
              {showTraceDetails ? 'ซ่อนรายละเอียด Trace ↑' : 'ดูรายละเอียด Trace →'}
            </button>
            {showTraceDetails ? (
              <div className="mt-4 rounded-2xl border border-white/10 bg-[#0B1126] p-4 space-y-3">
                {traceSteps.length === 0 ? (
                  <div className="text-sm text-gray-400">
                    ยังไม่มีข้อมูล Trace
                  </div>
                ) : (
                  traceSteps.map((step, index) => (
                    <div
                      key={`detail-${step.step}-${index}`}
                      className="rounded-xl border border-white/10 bg-white/5 p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-semibold text-violet-200">
                          Step {index + 1}: {step.step}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>{formatDuration(step.duration_ms)}</span>
                          <span>{step.status}</span>
                        </div>
                      </div>

                      <div className="text-xs text-gray-300 leading-6">
                        <div>
                          <span className="text-gray-500">Reasoning:</span>{' '}
                          {step.reasoning || '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Feedback:</span>{' '}
                          {step.feedback || '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Top Score:</span>{' '}
                          {step.top_score ?? '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Duration:</span>{' '}
                          {formatDuration(step.duration_ms)}
                        </div>
                        <div>
                          <span className="text-gray-500">Breakdown:</span>{' '}
                          {formatTraceBreakdown(step.duration_breakdown) || '-'}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : null}
          </div>
        ) : (
          <>
            <div
              className="relative rounded-3xl border border-white/10 bg-[#0A1023] overflow-hidden p-8 mb-6"
              style={{
                backgroundImage:
                  'radial-gradient(circle at 20% 20%, rgba(124, 58, 237, 0.25), transparent 45%), radial-gradient(circle at 80% 10%, rgba(236, 72, 153, 0.2), transparent 40%), linear-gradient(120deg, rgba(8, 12, 29, 0.95), rgba(12, 18, 40, 0.95))',
              }}
            >
              <div className="absolute inset-0 opacity-40" style={{
                backgroundImage:
                  'radial-gradient(rgba(255, 255, 255, 0.12) 1px, transparent 1px)',
                backgroundSize: '24px 24px',
              }} />

              <div className="relative z-10 flex items-center gap-8">
                <div className="flex-1">
                  <h1 className="text-4xl md:text-5xl font-black leading-tight">
                    Discover Anime <span className="text-violet-300">OST</span>
                    <br />
                    Through <span className="text-fuchsia-300">AI Agents</span>
                  </h1>
                  <p className="text-gray-300 mt-4 max-w-2xl">
                    บอกความรู้สึกของคุณ แล้วให้ AI ช่วยค้นหาเพลงประกอบที่ใช่
                  </p>

                  <div className="mt-6 flex flex-wrap gap-3">
                    {[
                      'เศร้า เหงา อบอุ่น',
                      'ต่อสู้ มั่นใจ เร้าใจ',
                      'ผ่อนคลาย ฟังสบาย',
                      'ลึกลับ น่าค้นหา',
                      'เปิดตอนทำงาน / อ่านหนังสือ',
                    ].map((chip) => (
                      <span
                        key={chip}
                        className="px-4 py-2 rounded-full text-xs text-gray-200 border border-white/10 bg-white/5"
                      >
                        {chip}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="hidden lg:block w-[320px] h-[260px] rounded-3xl border border-white/10 bg-gradient-to-br from-violet-500/20 via-transparent to-fuchsia-500/20 relative">
                  <div className="absolute inset-0 rounded-3xl bg-gradient-to-t from-[#0A1023] via-transparent to-transparent" />
                  <div className="absolute -right-10 -top-10 w-52 h-52 rounded-full bg-violet-500/30 blur-3xl" />
                  <div className="absolute bottom-6 right-6 w-16 h-16 rounded-full border border-white/20 bg-black/40 flex items-center justify-center">
                    ♪
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
            <div className="rounded-3xl border border-white/10 bg-[#0A1023] p-6 shadow-[0_0_40px_rgba(88,101,242,0.08)]">
              <div className="flex items-center gap-2 text-sm text-violet-200 mb-5">
                <span className="text-violet-300">✦</span>
                <span className="font-semibold">AI Chat</span>
              </div>

              <div className="space-y-5">
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-xs font-semibold">
                    YOU
                  </div>
                  <div className="flex-1 rounded-2xl bg-white/5 border border-white/10 px-4 py-3 text-sm text-gray-200">
                    {lastUserMessage || 'ลองถามหาเพลงอนิเมะที่คุณอยากฟัง'}
                  </div>
                  <span className="text-xs text-gray-500">12:42</span>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-full bg-[#11172c] border border-violet-500/40 flex items-center justify-center text-violet-300 text-sm">
                    AI
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-semibold text-gray-200 mb-1">Scout Agent</div>
                    <div className="rounded-2xl bg-white/5 border border-white/10 px-4 py-3 text-sm text-gray-300 whitespace-pre-line">
                      {agentAnswer || 'AI Agent พร้อมช่วยค้นหา Anime OST'}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4">
                  {
                    results.map((item) => (
                    <div
                      key={item.title}
                      className="rounded-2xl border border-white/10 bg-[#0B1126] overflow-hidden"
                    >
                      <div className="h-24 bg-gradient-to-br from-violet-500/40 to-fuchsia-500/30 relative overflow-hidden">
                        {item.imageUrl ? (
                          <img
                            src={item.imageUrl}
                            alt={item.title}
                            className="absolute inset-0 h-full w-full object-cover"
                            onError={(event) => {
                              event.currentTarget.style.display = 'none'
                            }}
                          />
                        ) : null}
                        <a
                          href={item.href}
                          target="_blank"
                          rel="noreferrer"
                          className="absolute bottom-3 right-3 w-9 h-9 rounded-full bg-black/50 border border-white/20 flex items-center justify-center"
                        >
                          ►
                        </a>
                      </div>
                      <div className="p-3">
                        <div className="text-xs font-semibold leading-snug text-gray-100">
                          {item.title}
                        </div>
                        <div className="text-xs text-emerald-400 mt-2">
                          {item.match}% match
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="rounded-2xl border border-white/10 bg-[#0B1126] p-2 flex items-center">
                  <input
                    placeholder="Ask anything about anime OST..."
                    value={chatInput}
                    onChange={handleChatInputChange}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        handleChatSubmit()
                      }
                    }}
                    disabled={isLoading}
                    className="flex-1 bg-transparent px-4 py-3 text-sm outline-none"
                  />
                  <button
                    type="button"
                    onClick={handleChatSubmit}
                    disabled={isLoading}
                    className="w-10 h-10 rounded-xl bg-violet-600 disabled:opacity-60"
                  >
                    ➤
                  </button>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-[#0A1023] p-6 shadow-[0_0_40px_rgba(88,101,242,0.08)]">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2 text-sm text-violet-200">
                  <span className="text-violet-300">✦</span>
                  <span className="font-semibold">Agent Trace</span>
                  <span className="ml-2 px-2 py-0.5 rounded-full bg-violet-500/20 text-[10px] text-violet-200">
                    LIVE
                  </span>
                </div>
                <button type="button" className="text-xs text-gray-400 hover:text-gray-200">
                  ⤴
                </button>
              </div>

              <div className="space-y-4">
                {traceSteps.map((step, index) => (
                  <div key={`${step.step}-${index}`} className="flex items-center gap-4">
                    <div
                      className={`w-10 h-10 rounded-full text-xs font-semibold flex items-center justify-center text-[#050816]
                      ${
                        step.status === 'success'
                          ? 'bg-gradient-to-br from-emerald-500 to-lime-300'
                          : 'bg-gradient-to-br from-violet-500 to-fuchsia-300'
                      }`}
                    >
                      {index + 1}
                    </div>

                    <div className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 flex items-center justify-between gap-4">
                      <div>
                        <div className="text-sm font-semibold text-gray-100">
                          {step.step}
                        </div>

                        <div className="text-xs text-gray-400 mt-1">
                          {step.reasoning || step.feedback || 'Agent processing'}
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-1 text-xs text-gray-400">
                        <span>{step.status}</span>
                        <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-violet-200">
                          {formatDuration(step.duration_ms)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
                {isTraceAnimating ? (
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-300 animate-pulse" />

                    <div className="flex-1 rounded-2xl border border-violet-500/30 bg-violet-500/10 px-4 py-3">
                      <div className="text-sm font-semibold text-violet-200">
                        Agent is thinking...
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        กำลังประมวลผลขั้นตอนถัดไป
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              <button
                type="button"
                onClick={() => setShowTraceDetails((prev) => !prev)}
                className="mt-6 w-full rounded-2xl border border-violet-500/30 bg-violet-500/10 py-3 text-sm text-violet-200 hover:bg-violet-500/20 transition-colors"
              >
                {showTraceDetails ? 'ซ่อนรายละเอียด Trace ↑' : 'ดูรายละเอียด Trace →'}
              </button>
              {showTraceDetails ? (
              <div className="mt-4 rounded-2xl border border-white/10 bg-[#0B1126] p-4 space-y-3">
                {traceSteps.length === 0 ? (
                  <div className="text-sm text-gray-400">
                    ยังไม่มีข้อมูล Trace
                  </div>
                ) : (
                  traceSteps.map((step, index) => (
                    <div
                      key={`detail-${step.step}-${index}`}
                      className="rounded-xl border border-white/10 bg-white/5 p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-semibold text-violet-200">
                          Step {index + 1}: {step.step}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>{formatDuration(step.duration_ms)}</span>
                          <span>{step.status}</span>
                        </div>
                      </div>

                      <div className="text-xs text-gray-300 leading-6">
                        <div>
                          <span className="text-gray-500">Reasoning:</span>{' '}
                          {step.reasoning || '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Feedback:</span>{' '}
                          {step.feedback || '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Top Score:</span>{' '}
                          {step.top_score ?? '-'}
                        </div>
                        <div>
                          <span className="text-gray-500">Duration:</span>{' '}
                          {formatDuration(step.duration_ms)}
                        </div>
                        <div>
                          <span className="text-gray-500">Breakdown:</span>{' '}
                          {formatTraceBreakdown(step.duration_breakdown) || '-'}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : null}
            </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
