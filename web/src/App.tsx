import { useState, useCallback } from 'react'

// Basic types for the API responses
interface SearchResult {
  chunk_id: string;
  document_id: string;
  text: string;
  metadata: Record<string, any>;
  metadata_str?: string;
  source: string;
  item_key: string;
  score: number;
}

interface QAResult {
  status: string;
  item_key: string;
  fulltext?: string;
  details?: Record<string, any>;
  relevant_chunks?: SearchResult[];
  error?: string;
}

const SearchIcon = () => (
  <svg className="w-5 h-5 text-google-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
)

const BookIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
)

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'search' | 'qa'>('search')
  const [itemKey, setItemKey] = useState('')
  const [loading, setLoading] = useState(false)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [qaResult, setQaResult] = useState<QAResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query.trim() && mode === 'search') return;
    if (!itemKey.trim() && mode === 'qa') return;

    setLoading(true)
    setError(null)
    setSearchResults([])
    setQaResult(null)

    try {
      if (mode === 'search') {
        const response = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, limit: 10 }),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json()
        setSearchResults(data.results || [])
      } else {
        const response = await fetch('/api/qa', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ item_key: itemKey, question: query }),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json()
        setQaResult(data)
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during the request.')
    } finally {
      setLoading(false)
    }
  }, [query, itemKey, mode])

  return (
    <div className="min-h-screen bg-google-gray-100/50 flex flex-col items-center">
      {/* Header Container */}
      <header className={`w-full max-w-4xl px-6 py-8 flex flex-col transition-all duration-500 ease-in-out ${(searchResults.length === 0 && !qaResult && !loading) ? 'mt-32 items-center' : 'mt-4 items-start'}`}>
        <div className="flex items-center gap-3 mb-8">
          <img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" alt="Google" className="h-8" />
          <span className="text-2xl font-medium text-google-gray-700 tracking-tight">Literature</span>
        </div>

        {/* Search Input Container */}
        <div className="w-full relative shadow-sm rounded-full bg-white flex items-center pr-3 border border-google-gray-300 hover:shadow-google-md focus-within:shadow-google-lg transition-shadow duration-200">
          <div className="pl-5 pr-2 py-4">
            <SearchIcon />
          </div>
          <form className="flex-1" onSubmit={handleSearch}>
            {mode === 'search' ? (
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search literature (e.g. attention mechanisms)"
                className="w-full bg-transparent text-lg text-google-gray-900 outline-none placeholder:text-google-gray-400 py-3"
                autoFocus
              />
            ) : (
              <div className="flex w-full gap-2 py-2">
                <input
                  type="text"
                  value={itemKey}
                  onChange={(e) => setItemKey(e.target.value)}
                  placeholder="Zotero Item Key (e.g. AB12CD34)"
                  className="w-1/3 bg-transparent text-lg text-google-gray-900 outline-none border-r border-google-gray-300 px-2 placeholder:text-google-gray-400"
                />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Question (optional)"
                  className="flex-1 bg-transparent text-lg text-google-gray-900 outline-none px-2 placeholder:text-google-gray-400"
                />
              </div>
            )}
            <button type="submit" className="hidden">Search</button>
          </form>

          <button
            type="button"
            onClick={handleSearch}
            className="p-2 mr-1 rounded-full text-google-blue hover:bg-blue-50 transition-colors"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z" /></svg>
          </button>
        </div>

        {/* Mode Toggles */}
        <div className="flex gap-4 mt-6 ml-2 justify-center w-full max-w-xl">
          <button
            type="button"
            onClick={() => { setMode('search'); setError(null); }}
            className={`google-btn ${mode === 'search' ? 'bg-google-gray-100 text-google-gray-900 font-medium border border-google-gray-400' : 'bg-transparent text-google-gray-700 hover:bg-google-gray-100 border border-transparent'}`}
          >
            Semantic Search
          </button>
          <button
            type="button"
            onClick={() => { setMode('qa'); setError(null); }}
            className={`google-btn ${mode === 'qa' ? 'bg-google-gray-100 text-google-gray-900 font-medium border border-google-gray-400' : 'bg-transparent text-google-gray-700 hover:bg-google-gray-100 border border-transparent'}`}
          >
            Item Q&A lookup
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="w-full max-w-4xl px-6 pb-20 mt-4 flex-1">
        {loading && (
          <div className="mt-12 flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-google-blue"></div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded m-4">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Search Results */}
        {!loading && searchResults.length > 0 && mode === 'search' && (
          <div className="flex flex-col gap-5 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h2 className="text-sm text-google-gray-700 mb-2">About {searchResults.length} results</h2>
            {searchResults.map((result, idx) => (
              <div key={idx} className="google-card group">
                {/* Meta and Citation info */}
                <div className="flex flex-col mb-1">
                  <div className="flex items-center gap-2 text-sm text-google-gray-700 mb-1">
                    <BookIcon />
                    <span className="truncate max-w-md">{result.metadata?.item_key || result.item_key}</span>
                  </div>
                  {/* Highlighted Title or Context */}
                  <h3 className="text-xl text-[1.25rem] font-medium text-google-blue hover:underline cursor-pointer">
                    {result.metadata?.title || 'Document Section'}
                  </h3>
                </div>

                <p className="text-sm text-google-gray-800 leading-snug line-clamp-3">
                  {result.text}
                </p>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-google-gray-100 text-google-gray-800">
                    Relevance: {result.score?.toFixed(3)}
                  </span>
                  {result.metadata?.year && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-google-gray-100 text-google-gray-800">
                      Year: {result.metadata.year}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* QA Result */}
        {!loading && qaResult && mode === 'qa' && (
          <div className="flex flex-col gap-5 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-white rounded-2xl border border-google-gray-300 p-8 shadow-google-sm">
              <h2 className="text-2xl font-medium text-google-gray-900 mb-2">
                {qaResult.details?.title || `Item ${qaResult.item_key}`}
              </h2>
              {qaResult.details?.abstractText && (
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-google-gray-700 uppercase tracking-wider mb-2">Abstract</h4>
                  <p className="text-google-gray-800 text-sm leading-relaxed">{qaResult.details.abstractText}</p>
                </div>
              )}

              {qaResult.relevant_chunks && qaResult.relevant_chunks.length > 0 && (
                <div className="mt-8 border-t border-google-gray-200 pt-6">
                  <h3 className="text-lg font-medium text-google-gray-900 mb-4">Relevant Excerpts</h3>
                  <div className="flex flex-col gap-4">
                    {qaResult.relevant_chunks.map((chunk, idx) => (
                      <div key={idx} className="bg-google-gray-50 rounded-lg p-4 border border-google-gray-200">
                        <p className="text-sm text-google-gray-800 italic">...{chunk.text}...</p>
                        <div className="mt-2 flex justify-end">
                          <span className="text-xs text-google-gray-500">Score: {chunk.score.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!qaResult.relevant_chunks?.length && qaResult.fulltext && (
                <div className="mt-8 border-t border-google-gray-200 pt-6">
                  <h3 className="text-lg font-medium text-google-gray-900 mb-4">Full Document Text Snippet</h3>
                  <p className="text-sm text-google-gray-800 whitespace-pre-wrap font-mono bg-google-gray-50 p-4 rounded h-64 overflow-y-auto">
                    {qaResult.fulltext.slice(0, 5000)}...
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
