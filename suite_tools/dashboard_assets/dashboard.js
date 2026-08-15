    const pollMs = Number(document.body.dataset.pollMs || 2500);
    const csrfToken = document.body.dataset.csrfToken || ''; // # noqa: release-audit-fixture
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    const app = document.getElementById('app');
    const lastRefresh = document.getElementById('lastRefresh');
    const liveLabel = document.getElementById('liveLabel');
    const copyPanel = document.getElementById('copyPanel');
    const copyPanelTitle = document.getElementById('copyPanelTitle');
    const copyPanelNote = document.getElementById('copyPanelNote');
    const copyTextarea = document.getElementById('copyTextarea');
    const closeCopyPanel = document.getElementById('closeCopyPanel');
    const selectCopyText = document.getElementById('selectCopyText');
    const themeToggle = document.getElementById('themeToggle');
    const themeLabel = document.getElementById('themeLabel');
    const themeGlyph = document.getElementById('themeGlyph');
    const brandShield = document.getElementById('brandShield');
    const topScopeControl = document.getElementById('topScopeControl');
    const topComplete = document.getElementById('topComplete');
    const topElapsed = document.getElementById('topElapsed');
    const topElapsedLabel = document.getElementById('topElapsedLabel');
    const topActive = document.getElementById('topActive');
    const topErrors = document.getElementById('topErrors');
    const topErrorsStat = document.getElementById('topErrorsStat');
    const topActiveStat = document.getElementById('topActiveStat');
    const THEME_STORAGE_KEY = 'benchmarkDashboardTheme';
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const money = (value) => `$${Number(value || 0).toFixed(4)}`;
    const moneyCents = (value) => `$${Number(value || 0).toFixed(2)}`;
    const shortHash = (value) => {
      const text = String(value ?? '');
      if (!text) return 'unknown';
      return text.length > 22 ? `${text.slice(0, 12)}...${text.slice(-6)}` : text;
    };
    const seenEvents = new Set();
    let freshEvents = new Set();
    let firstPaint = true;
    let firstEvidencePaint = true;
    let activeFilter = 'all';
	    let activeStageFilter = 'all';
	    let lastData = null;
	    let lastAppHtml = '';
	    let runsEtag = '';
	    let refreshTimer = null;
	    let refreshInFlight = false;
	    let scopeRefreshPending = false;
	    let evidenceRequestKey = '';
	    let evidenceRequestSequence = 0;
	    let contractRequestKey = '';
	    let contractRequestSequence = 0;
	    const detailEtags = new Map();
	    const detailCache = new Map();
	    let showAcknowledged = false;
	    const DEFAULT_EVIDENCE_TRACE_WINDOW = '100';
	    let evidenceAutoFollow = true;
	    let evidenceTraceAutoFollow = true;
	    let evidenceTraceWindow = DEFAULT_EVIDENCE_TRACE_WINDOW;
	    let suppressFreshOnNextRender = false;
	    let evidenceRunScope = 'workflow:active';
		    let evidenceContentFilter = 'all';
		    let selectedEvidenceKey = '';
		    let evidenceTracePoints = [];
		    const evidenceTraceBirths = new Map();
		    const EVIDENCE_TRACE_ENTER_MS = 420;
		    const EVIDENCE_TRACE_STAGGER_MS = 24;
		    let evidenceTraceAnimationFrame = 0;
		    let lastEvidenceViewSignature = '';
	    let lastEvidenceRunFingerprint = '';
		    let pendingFeedPanelScroll = false;
	    let pendingEvidenceLiveSnap = false;
		    let modelRegistry = new Map();
		    const openDetails = new Set();
		    const queueExpansionState = new Map();
		    const previousQueueStageSignatures = new Map();
		    const seenQueueGroupKeys = new Set();
    const ACK_STORAGE_KEY = 'benchmarkDashboardAcknowledged.v1';
    const acknowledged = new Set();
    const filters = [['all', 'All'], ['running', 'Running'], ['attention', 'Attention'], ['ready', 'Ready']];

    function replaceAppHtml(html) {
      app.innerHTML = html;
      lastAppHtml = html;
      app.dataset.paintCount = String(Number(app.dataset.paintCount || 0) + 1);
    }

    const RUN_BUILDER_MODULES = [
	      {key: 'aita', label: 'AITA', note: 'social pressure pairs'},
      {key: 'epis', label: 'Epistemic', note: 'belief pressure'},
	      {key: 'sus', label: 'SUS', note: 'unsafe-suggestion pressure'}
    ];
    const RUN_BUILDER_STAGES = [
      {key: 'validate', label: 'Validate', note: 'free'},
      {key: 'render', label: 'Render configs', note: 'free'},
      {key: 'prepare', label: 'Prepare contracts', note: 'no paid calls'},
      {key: 'schedule', label: 'Run scheduler', note: 'paid'}
    ];
    const RUN_BUILDER_SIZES = [
      {key: 'tiny', label: 'Tiny smoke', note: '1 item'},
      {key: 'sample', label: 'Sample', note: 'small slice'},
      {key: 'wide', label: 'Wide pass', note: 'larger slice'}
    ];
    const runBuilderState = {
      modelGroups: [],
      judgeSets: [],
      modules: ['aita'],
      stage: 'prepare',
      size: 'tiny',
      runIdPrefix: 'operator',
      outputRoot: 'results/testing',
      maxActiveCalls: '2',
    };
	    function noteStaticSnapshot() {
	      lastRefresh.textContent = `No ledger changes; static snapshot checked ${formatTime(new Date().toISOString())}`;
	      liveLabel.textContent = 'Static snapshot';
	      requestEvidenceTraceLiveScroll();
	    }

	    const brandMarks = {
	      openai: {label: 'OpenAI', path: 'M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z'},
	      anthropic: {label: 'Anthropic', path: 'M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z'},
	      gemini: {label: 'Google Gemini', path: 'M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81'},
	      google: {label: 'Google', path: 'M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z'},
	      xai: {label: 'xAI / Grok', path: 'M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 11.09h-3.182z'},
	      mistral: {label: 'Mistral AI', path: 'M17.143 3.429v3.428h-3.429v3.429h-3.428V6.857H6.857V3.43H3.43v13.714H0v3.428h10.286v-3.428H6.857v-3.429h3.429v3.429h3.429v-3.429h3.428v3.429h-3.428v3.428H24v-3.428h-3.43V3.429z'},
	      qwen: {label: 'Alibaba Cloud / Qwen', text: 'Q'},
	      zhipu: {label: 'Zhipu AI / GLM', text: 'Z'},
	      moonshot: {label: 'Moonshot AI / Kimi', text: 'K'},
	      deepseek: {label: 'DeepSeek', text: 'D'},
	      xiaomi: {label: 'Xiaomi / MiMo', text: 'mi'},
	      nvidia: {label: 'NVIDIA / Nemotron', text: 'N'},
	      therapeuticHarness: {label: 'Therapeutic Harness', text: 'TH'},
	    };

    function loadAcknowledged() {
      try {
        const values = JSON.parse(window.localStorage.getItem(ACK_STORAGE_KEY) || '[]');
        for (const value of values || []) acknowledged.add(String(value));
      } catch (error) {
        acknowledged.clear();
      }
    }

    function saveAcknowledged() {
      window.localStorage.setItem(ACK_STORAGE_KEY, JSON.stringify([...acknowledged].sort()));
    }

    function acknowledgeKey(value) {
      if (!value) return '';
      return String(value);
    }

    function moduleAckKey(module) {
      return acknowledgeKey(module.status_path || module.output_dir || [module.group, module.run_id, module.module_path || module.module].filter(Boolean).join('|'));
    }

    function flowAckKey(item) {
      return acknowledgeKey(item.status_path || item.output_dir || item.contract_path || [item.run_id, item.module_path || item.module].filter(Boolean).join('|'));
    }

    function isAcknowledgedModule(module) {
      const key = moduleAckKey(module);
      return key ? acknowledged.has(key) : false;
    }

    function isRejectedModule(module) {
      return (module.disposition || {}).disposition === 'rejected_from_analysis';
    }

    function isAcknowledgedFlowItem(item) {
      const key = flowAckKey(item);
      return key ? acknowledged.has(key) : false;
    }

    function isRejectedFlowItem(item) {
      return item.analysis_state === 'rejected_from_analysis' || item.lane === 'rejected';
    }

    function isHiddenModule(module) {
      return isRejectedModule(module) || isAcknowledgedModule(module);
    }

    function isHiddenFlowItem(item) {
      return isRejectedFlowItem(item) || isAcknowledgedFlowItem(item);
    }

	    function updateModelRegistry(data) {
	      const registry = new Map();
	      for (const model of ((data.operator || {}).models || [])) {
	        if (!model || typeof model !== 'object') continue;
	        const record = {
	          key: String(model.key || ''),
	          label: String(model.label || model.key || model.model_id || ''),
	          model_id: String(model.model_id || ''),
	          endpoint: String(model.endpoint || ''),
	        };
	        [record.key, record.label, record.model_id].filter(Boolean).forEach((value) => {
	          registry.set(value.toLowerCase(), record);
	        });
	      }
	      modelRegistry = registry;
	    }

	    function modelRecord(value) {
	      const raw = String(value || '').trim();
	      if (!raw) return null;
	      return modelRegistry.get(raw.toLowerCase()) || null;
	    }

	    function modelIdentityText(value) {
	      const record = modelRecord(value);
	      return [value, record?.label, record?.model_id, record?.key].filter(Boolean).join(' ');
	    }

	    function brandForModel(value) {
	      const model = modelIdentityText(value).toLowerCase();
	      if (model.includes('therapeutic-harness') || model.includes('therapeutic harness') || model.startsWith('th-') || model.includes(' th-')) return 'therapeuticHarness';
	      if (model.includes('openai') || model.includes('gpt') || model.includes('chatgpt') || model.includes('codex')) return 'openai';
	      if (model.includes('anthropic') || model.includes('claude') || model.includes('opus') || model.includes('sonnet') || model.includes('haiku')) return 'anthropic';
	      if (model.includes('gemini')) return 'gemini';
	      if (model.includes('google') || model.includes('gemma')) return 'google';
	      if (model.includes('grok') || model.includes('x-ai')) return 'xai';
	      if (model.includes('mistral')) return 'mistral';
	      if (model.includes('deepseek')) return 'deepseek';
	      if (model.includes('qwen')) return 'qwen';
	      if (model.includes('glm') || model.includes('z-ai') || model.includes('zhipu')) return 'zhipu';
	      if (model.includes('kimi') || model.includes('moonshot')) return 'moonshot';
	      if (model.includes('mimo') || model.includes('xiaomi')) return 'xiaomi';
	      if (model.includes('nemotron') || model.includes('nvidia')) return 'nvidia';
	      return '';
	    }

	    function brandLogo(model) {
	      const brand = brandForModel(model);
	      const mark = brandMarks[brand];
	      if (!mark) return '';
	      if (mark.path) {
	        return `<span class="brand-logo brand-logo-${esc(brand)}" title="${esc(mark.label)}" aria-hidden="true"><svg viewBox="${esc(mark.viewBox || '0 0 24 24')}" focusable="false"><path d="${mark.path}"></path></svg></span>`;
	      }
	      return `<span class="brand-logo brand-logo-${esc(brand)}" title="${esc(mark.label)}" aria-hidden="true"><span class="brand-text">${esc(mark.text || '?')}</span></span>`;
	    }

	    function modelShortCode(value) {
	      const raw = String(value || '');
	      const record = modelRecord(raw);
	      const text = [raw, record?.label, record?.model_id, record?.key].filter(Boolean).join(' ');
	      const model = text.toLowerCase();
	      const version = (...patterns) => {
	        for (const pattern of patterns) {
	          const match = model.match(pattern);
	          if (match) return match[1].replace(/-/g, '.').replace(/\.0$/, '');
	        }
	        return '';
	      };
	      const codeWithVersion = (prefix, ...patterns) => {
	        const v = version(...patterns);
	        return v ? `${prefix}-${v}` : prefix;
	      };
	      const tokenPresent = (token) => new RegExp(`(^|[-_/\s()])${token}($|[-_/\s()])`).test(model);
	      const fallback = () => {
	        const words = raw.replace(/[-_/]+/g, ' ').split(/\s+/).filter(Boolean);
	        const letters = words.map((word) => word[0]).join('').toUpperCase();
	        const digits = raw.match(/\d+(?:\.\d+)?/);
	        return `${letters.slice(0, 2) || 'M'}${digits ? digits[0] : ''}`.slice(0, 8);
	      };
	      const baseCode = () => {
	        if (model.includes('opus')) return codeWithVersion('C-O', /opus\s*(\d+(?:[.-]\d+)?)/, /opus[-_/](\d+(?:[.-]\d+)?)/);
	        if (model.includes('sonnet')) return codeWithVersion('C-S', /sonnet\s*(\d+(?:[.-]\d+)?)/, /sonnet[-_/](\d+(?:[.-]\d+)?)/);
	        if (model.includes('haiku')) return codeWithVersion('C-H', /haiku\s*(\d+(?:[.-]\d+)?)/, /haiku[-_/](\d+(?:[.-]\d+)?)/);
	        if (model.includes('gpt') || model.includes('openai') || model.includes('chatgpt') || model.includes('codex')) {
	          const v = version(/gpt[-_\s]*(\d+(?:[.-]\d+)?[a-z]?)/, /chatgpt[-_\s]*(\d+(?:[.-]\d+)?[a-z]?)/, /codex[-_\s]*(\d+(?:[.-]\d+)?[a-z]?)/);
	          const size = tokenPresent('mini') ? '-mini' : (tokenPresent('nano') ? '-nano' : (tokenPresent('pro') ? '-pro' : ''));
	          const latest = model.includes('latest') ? '-latest' : '';
	          if (model.includes('codex')) return `CX${v ? '-' + v : ''}`;
	          if (model.includes('chatgpt') || model.includes('chat-gpt') || model.includes('chat latest') || model.includes('chat-latest')) return `CG${v ? '-' + v : ''}${latest}`;
	          return `GPT${v ? '-' + v : ''}${size}`;
	        }
	        if (model.includes('gemini')) {
	          const v = version(/gemini\s*(\d+(?:[.-]\d+)?)/, /gemini[-_/](\d+(?:[.-]\d+)?)/);
	          if (model.includes('flash-lite') || model.includes('flash lite')) return `G-FL${v ? '-' + v : ''}`;
	          if (model.includes('flash')) return `G-F${v ? '-' + v : ''}`;
	          if (model.includes('pro')) return `G-P${v ? '-' + v : ''}`;
	          return `G-G${v ? '-' + v : ''}`;
	        }
	        if (model.includes('kimi') || model.includes('moonshot')) return codeWithVersion('K', /k(?:imi)?\s*k?(\d+(?:[.-]\d+)?)/, /k(?:imi)?[-_]k?(\d+(?:[.-]\d+)?)/);
	        if (model.includes('glm')) return codeWithVersion('GLM', /glm\s*(\d+(?:[.-]\d+)?)/, /glm[-_](\d+(?:[.-]\d+)?)/);
	        if (model.includes('qwen')) return codeWithVersion('Q', /qwen\s*(\d+(?:[.-]\d+)?\+?)/, /qwen[-_](\d+(?:[.-]\d+)?\+?)/);
	        if (model.includes('deepseek')) return codeWithVersion('DS', /(?:v|r)(\d+(?:[.-]\d+)?)/, /deepseek\s*(\d+(?:[.-]\d+)?)/);
	        if (model.includes('grok')) return codeWithVersion('X', /grok\s*(\d+(?:[.-]\d+)?)/, /grok[-_](\d+(?:[.-]\d+)?)/);
	        if (model.includes('mimo')) return codeWithVersion('MI', /(?:v)?(\d+(?:[.-]\d+)?)/);
	        if (model.includes('nemotron')) return codeWithVersion('N', /nemotron\s*(\d+(?:[.-]\d+)?)/, /nemotron[-_](\d+(?:[.-]\d+)?)/);
	        if (model.includes('mistral')) return model.includes('large') ? 'M-L' : fallback();
	        if (model.includes('gemma')) return codeWithVersion('Gm', /gemma\s*(\d+(?:[.-]\d+)?)/, /gemma[-_](\d+(?:[.-]\d+)?)/);
	        return fallback();
	      };
	      const isHarness = model.includes('therapeutic-harness') || model.includes('therapeutic harness') || model.startsWith('th-') || model.includes(' th-');
	      if (isHarness) {
	        if (model.includes('gpt') || model.includes('openai')) return `TH-${baseCode()}`;
	        if (model.includes('gemini') || model.includes('flash')) return `TH-${baseCode().replace(/^G-/, 'Gemini-')}`;
	        if (model.includes('opus')) return `TH-${baseCode().replace(/^C-O/, 'Opus')}`;
	        if (model.includes('sonnet')) return `TH-${baseCode().replace(/^C-S/, 'Sonnet')}`;
	        return 'TH';
	      }
	      return baseCode();
	    }

	    function titleCaseModel(value) {
	      return String(value || '')
	        .replace(/^[a-z]+\/+/i, '')
	        .replace(/[-_]+/g, ' ')
	        .replace(/(\d)\s+(\d)/g, '$1.$2')
	        .replace(/\b(gpt|glm|qwen|gemini|claude|opus|sonnet|haiku|flash|pro|preview|native|high|xhigh|openrouter|codex|chatgpt|th|harness|guards|alpha)\b/gi, (word) => {
	          const key = word.toLowerCase();
	          const map = {gpt: 'GPT', glm: 'GLM', qwen: 'Qwen', gemini: 'Gemini', claude: 'Claude', opus: 'Opus', sonnet: 'Sonnet', haiku: 'Haiku', flash: 'Flash', pro: 'Pro', preview: 'Preview', native: 'Native', high: 'High', xhigh: 'XHigh', openrouter: 'OpenRouter', codex: 'Codex', chatgpt: 'ChatGPT', th: 'TH', harness: 'Harness', guards: 'Guards', alpha: 'Alpha'};
	          return map[key] || word;
	        })
	        .trim();
	    }

	    function modelDisplayParts(value) {
	      const raw = String(value || '').trim();
	      const record = modelRecord(raw);
	      const display = record?.label || raw;
	      const parts = display.split(' / ').map((part) => part.trim()).filter(Boolean);
	      if (parts.length > 1) return {name: titleCaseModel(parts[0]), condition: parts.slice(1).join(' / '), raw, record};
	      return {name: titleCaseModel(display), condition: '', raw, record};
	    }

	    function renderModelChip(value, mode = '') {
	      const raw = String(value || '').trim();
	      if (!raw) return '';
	      const parts = modelDisplayParts(raw);
	      const code = modelShortCode(raw);
	      const brand = brandForModel(raw);
	      const full = [parts.name, parts.condition, parts.record?.model_id && parts.record.model_id !== raw ? parts.record.model_id : '', code ? `code ${code}` : ''].filter(Boolean).join(' · ');
	      return `
	        <span class="model-chip ${brand ? `model-chip-${esc(brand)}` : ''} ${esc(mode)}" title="${esc(full)}">
	          ${brandLogo(raw)}
	          <span class="model-chip-name">${esc(parts.name || raw)}</span>
	          ${code ? `<span class="model-chip-code">${esc(code)}</span>` : ''}
	        </span>
	      `;
	    }

	    function renderModelStack(models, limit = 4) {
	      const values = (models || []).filter(Boolean);
	      const visible = values.slice(0, limit).map((model) => renderModelChip(model, 'compact')).join('');
	      const more = values.length > limit ? `<span class="chip">+${esc(values.length - limit)}</span>` : '';
	      return visible || more ? `<div class="model-stack">${visible}${more}</div>` : '';
	    }

	    function inlineMarkdown(value) {
	      return esc(value)
	        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
	        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
	        .replace(/__(.+?)__/g, '<strong>$1</strong>')
	        .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>')
	        .replace(/(^|[^_])_([^_]+)_/g, '$1<em>$2</em>')
	        .replace(/`([^`]+)`/g, '<code>$1</code>');
	    }

	    function renderMarkdown(raw) {
	      const lines = String(raw ?? '').replace(/\r\n/g, '\n').split('\n');
	      const blocks = [];
	      let paragraph = [];
	      let listType = '';
	      let listItems = [];
	      let codeLines = null;
	      const flushParagraph = () => {
	        if (!paragraph.length) return;
	        blocks.push(`<p>${paragraph.map(inlineMarkdown).join('<br>')}</p>`);
	        paragraph = [];
	      };
	      const flushList = () => {
	        if (!listItems.length) return;
	        blocks.push(`<${listType}>${listItems.map((item) => `<li>${item}</li>`).join('')}</${listType}>`);
	        listType = '';
	        listItems = [];
	      };
	      const flushCode = () => {
	        if (!codeLines) return;
	        blocks.push(`<pre><code>${esc(codeLines.join('\n'))}</code></pre>`);
	        codeLines = null;
	      };
	      for (const line of lines) {
	        const trimmed = line.trim();
	        if (trimmed.startsWith('```')) {
	          if (codeLines) flushCode();
	          else {
	            flushParagraph();
	            flushList();
	            codeLines = [];
	          }
	          continue;
	        }
	        if (codeLines) {
	          codeLines.push(line);
	          continue;
	        }
	        if (!trimmed) {
	          flushParagraph();
	          flushList();
	          continue;
	        }
	        const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
	        if (heading) {
	          flushParagraph();
	          flushList();
	          const level = Math.min(heading[1].length, 4);
	          blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
	          continue;
	        }
	        const quote = trimmed.match(/^>\s+(.+)$/);
	        if (quote) {
	          flushParagraph();
	          flushList();
	          blocks.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
	          continue;
	        }
	        const unordered = trimmed.match(/^[-*]\s+(.+)$/);
	        if (unordered) {
	          flushParagraph();
	          if (listType && listType !== 'ul') flushList();
	          listType = 'ul';
	          listItems.push(inlineMarkdown(unordered[1]));
	          continue;
	        }
	        const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
	        if (ordered) {
	          flushParagraph();
	          if (listType && listType !== 'ol') flushList();
	          listType = 'ol';
	          listItems.push(inlineMarkdown(ordered[1]));
	          continue;
	        }
	        flushList();
	        paragraph.push(trimmed);
	      }
	      flushParagraph();
	      flushList();
	      flushCode();
	      return blocks.join('') || '<p></p>';
	    }

	    function unwrapMessageContent(content) {
	      const text = String(content ?? '');
	      const trimmed = text.trim();
	      if (trimmed.startsWith('{') && trimmed.endsWith('}') && trimmed.includes('"response"')) {
	        try {
	          const parsed = JSON.parse(trimmed);
	          if (typeof parsed.response === 'string') return parsed.response;
	        } catch (error) {
	          return text;
	        }
	      }
	      return text;
	    }

    function rawModules(data) {
      return (data.groups || []).flatMap((group) =>
        (group.modules || []).map((module) => ({...module, run_id: group.run_id}))
      );
    }

    loadAcknowledged();

    function setTheme(theme, persist) {
      const safeTheme = theme === 'dark' ? 'dark' : 'light';
      document.documentElement.dataset.theme = safeTheme;
      document.documentElement.classList.toggle('dark', safeTheme === 'dark');
      if (persist) {
        try {
          window.localStorage.setItem(THEME_STORAGE_KEY, safeTheme);
        } catch (error) {}
      }
      if (themeGlyph) themeGlyph.textContent = safeTheme === 'dark' ? '◑' : '◐';
      if (themeLabel) {
        const stored = (() => {
          try { return window.localStorage.getItem(THEME_STORAGE_KEY); }
          catch (error) { return null; }
        })();
        themeLabel.textContent = stored
          ? `Theme set to ${safeTheme}`
          : `Theme follows system: ${safeTheme}`;
      }
      if (themeToggle) {
        themeToggle.setAttribute('aria-label', `Switch to ${safeTheme === 'dark' ? 'light' : 'dark'} theme`);
        themeToggle.title = `Switch to ${safeTheme === 'dark' ? 'light' : 'dark'} theme`;
      }
      window.requestAnimationFrame(() => drawEvidenceTrace(lastData));
    }

    function systemTheme() {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function storedTheme() {
      try {
        return window.localStorage.getItem(THEME_STORAGE_KEY);
      } catch (error) {
        return null;
      }
    }

    setTheme(storedTheme() || systemTheme(), false);

    function eventKey(event) {
      return [event.group, event.module_path, event.sequence ?? event.timestamp ?? '', event.event ?? ''].join('|');
    }

    function moduleKey(module) {
      return [module.group, module.module_path || module.module || 'module'].join('|');
    }

    function failureCopy(module) {
      const attention = module.attention || {};
      const examples = (attention.incomplete_examples || []).map((item) => `- ${item}`).join('\n');
      return [
        `${module.group || ''} / ${module.module_path || module.module || ''}`,
        `${module.status || 'unknown'} / ${module.validity || 'unknown'}`,
        module.score_state ? `Score state: ${module.score_state.label} - ${module.score_state.action}` : '',
        attention.reason ? `Reason: ${attention.reason}` : '',
        examples ? `Examples:\n${examples}` : '',
        module.status_path ? `Status: ${module.status_path}` : '',
        module.events_path ? `Events: ${module.events_path}` : '',
      ].filter(Boolean).join('\n');
    }

    function shellQuote(value) {
      return "'" + String(value ?? '').replace(/'/g, "'\''") + "'";
    }

    function runBuilderOptionLabel(options, key) {
      return (options.find((option) => option.key === key) || {}).label || key;
    }

    function runBuilderDefaults(operator) {
      const groups = (operator.model_groups || []).map((group) => group.name).filter(Boolean);
      const judges = (operator.judge_sets || []).map((judge) => judge.name).filter(Boolean);
      const defaultGroup = groups.includes('calibration_smoke') ? 'calibration_smoke' : (groups[0] || '');
      const defaultJudge = judges.includes('calibration') ? 'calibration' : (judges[0] || '');
      return {groups, judges, defaultGroup, defaultJudge};
    }

    function normalizeRunBuilderState(operator) {
      const defaults = runBuilderDefaults(operator);
      const moduleKeys = RUN_BUILDER_MODULES.map((module) => module.key);
      const stageKeys = RUN_BUILDER_STAGES.map((stage) => stage.key);
      const sizeKeys = RUN_BUILDER_SIZES.map((size) => size.key);
      const normalized = {
        modelGroups: runBuilderState.modelGroups.filter((name) => defaults.groups.includes(name)),
        judgeSets: runBuilderState.judgeSets.filter((name) => defaults.judges.includes(name)),
        modules: runBuilderState.modules.filter((name) => moduleKeys.includes(name)),
        stage: stageKeys.includes(runBuilderState.stage) ? runBuilderState.stage : 'prepare',
        size: sizeKeys.includes(runBuilderState.size) ? runBuilderState.size : 'tiny',
        runIdPrefix: String(runBuilderState.runIdPrefix || 'operator').trim() || 'operator',
        outputRoot: String(runBuilderState.outputRoot || 'results/testing').trim() || 'results/testing',
        maxActiveCalls: String(runBuilderState.maxActiveCalls || '2').trim() || '2',
      };
      if (!normalized.modelGroups.length && defaults.defaultGroup) normalized.modelGroups = [defaults.defaultGroup];
      if (!normalized.judgeSets.length && defaults.defaultJudge) normalized.judgeSets = [defaults.defaultJudge];
      if (!normalized.modules.length) normalized.modules = ['aita'];
      Object.assign(runBuilderState, normalized);
      return normalized;
    }

    function runBuilderFlags(size) {
      const flags = {
        tiny: {
          aita: '--items 1 --dataset-mode nta-paired',
          epis: '--items 1 --types delusion,pickside,mirror',
          sus: '--scenarios bridge_heights --runs 1',
        },
        sample: {
          aita: '--items 5 --dataset-mode nta-paired',
          epis: '--items 4 --types delusion,pickside,mirror',
          sus: '--runs 1',
        },
        wide: {
          aita: '--items 20 --dataset-mode nta-paired',
          epis: '--items 12 --types delusion,pickside,mirror',
          sus: '--runs 3',
        },
      };
      return flags[size] || flags.tiny;
    }

    function runBuilderShellArray(values) {
      return values.map(shellQuote).join(' ');
    }

    function buildRunBuilderOutput(operator, state = normalizeRunBuilderState(operator)) {
      const stage = state.stage || 'prepare';
      const selectors = state.modelGroups.map((group) => `group:${group}`);
      const flags = runBuilderFlags(state.size);
      const stageLabel = runBuilderOptionLabel(RUN_BUILDER_STAGES, stage);
      const sizeLabel = runBuilderOptionLabel(RUN_BUILDER_SIZES, state.size);
      const moduleNames = state.modules
        .map((key) => runBuilderOptionLabel(RUN_BUILDER_MODULES, key))
        .join(', ');
      const commandLines = [
        '# Anti-sycophancy benchmark run builder',
        '# Run this from the benchmark repo root: the directory containing suite_models.yaml.',
        'if [ ! -f suite_models.yaml ]; then',
        '  echo "Run this from the benchmark repo root: the directory containing suite_models.yaml." >&2',
        '  exit 2',
        'fi',
        '',
        `RUN_ID_PREFIX=${shellQuote(state.runIdPrefix)}`,
        `OUTPUT_ROOT=${shellQuote(state.outputRoot)}`,
        `MAX_ACTIVE_CALLS=${shellQuote(state.maxActiveCalls)}`,
        `BUILDER_STAGE=${shellQuote(stage)}`,
        `MODEL_GROUPS=(${runBuilderShellArray(selectors)})`,
        `JUDGE_SETS=(${runBuilderShellArray(state.judgeSets)})`,
        `MODULES=(${runBuilderShellArray(state.modules)})`,
        '',
        '# 1. Free validation and inventory.',
        './venv/bin/python -m suite_tools.model_config --validate',
        './venv/bin/python -m suite_tools.model_config --list',
        './venv/bin/python -m suite_tools.offline_gate',
        './venv/bin/python -m suite_tools.openrouter_preflight --config suite_models.yaml',
        '',
        'if [ "$BUILDER_STAGE" = "validate" ]; then exit 0; fi',
        '',
        '# 2. Free rendered configs for each selected model group and judge set.',
        'for MODEL_SELECTOR in "${MODEL_GROUPS[@]}"; do',
        '  for JUDGE_SET in "${JUDGE_SETS[@]}"; do',
        '    ./venv/bin/python -m suite_tools.model_config \\',
        '      --judge-set "$JUDGE_SET" \\',
        '      --models "$MODEL_SELECTOR" \\',
        '      --module all \\',
        '      --output-dir /tmp/benchmark-configs',
        '  done',
        'done',
        '',
        'if [ "$BUILDER_STAGE" = "render" ]; then exit 0; fi',
        '',
        '# 3. Prepare no-paid RUN_CONTRACT.json files. Inspect before scheduling.',
        'for MODEL_SELECTOR in "${MODEL_GROUPS[@]}"; do',
        '  SELECTOR_NAME="${MODEL_SELECTOR#group:}"',
        '  for JUDGE_SET in "${JUDGE_SETS[@]}"; do',
        '    for MODULE in "${MODULES[@]}"; do',
        '      case "$MODULE" in',
        `        aita) MODULE_FLAGS=(${flags.aita}) ;;`,
        `        epis) MODULE_FLAGS=(${flags.epis}) ;;`,
        `        sus) MODULE_FLAGS=(${flags.sus}) ;;`,
        '        *) echo "Unknown module: $MODULE" >&2; exit 2 ;;',
        '      esac',
        '      RUN_ID="$RUN_ID_PREFIX-$MODULE-$SELECTOR_NAME-$JUDGE_SET-$(date +%Y%m%d-%H%M%S)"',
        '      ./venv/bin/python -m suite_tools.prepare_run \\',
        '        --module "$MODULE" \\',
        '        --run-id "$RUN_ID" \\',
        '        --output "$OUTPUT_ROOT/$RUN_ID" \\',
        '        --models "$MODEL_SELECTOR" \\',
        '        --judge-set "$JUDGE_SET" \\',
        '        "${MODULE_FLAGS[@]}" \\',
        '        --non-interactive',
        '',
        '      if [ "$BUILDER_STAGE" = "schedule" ]; then',
        '        ./venv/bin/python -m suite_tools.scheduler run \\',
        '          --contract "$OUTPUT_ROOT/$RUN_ID/$MODULE/RUN_CONTRACT.json" \\',
        '          --max-active-calls "$MAX_ACTIVE_CALLS" \\',
        '          --auto-score-on-clean-generation \\',
        '          --stop-on-attention',
        '      fi',
        '    done',
        '  done',
        'done',
      ];
      const prompt = [
        'Use the benchmark operator workflow from the repository root that contains suite_models.yaml.',
        '',
        `Goal: ${stageLabel} for ${sizeLabel.toLowerCase()} across ${moduleNames || 'selected modules'}.`,
        `Model selectors: ${selectors.join(', ') || 'choose from suite_models.yaml'}.`,
        `Judge sets: ${state.judgeSets.join(', ') || 'choose from suite_models.yaml'}.`,
        `Output root: ${state.outputRoot}. Max active scheduler calls: ${state.maxActiveCalls}.`,
        '',
        'Rules:',
        '- Treat suite_models.yaml as the source of truth.',
        '- Run validation and preflight before any paid generation.',
        '- Use suite_tools.prepare_run to write no-paid RUN_CONTRACT.json files first.',
        '- Do not expose private routing, prompts, service ids, credentials, or ignored run artifacts.',
        stage === 'schedule'
          ? '- Only start the scheduler after confirming the prepared contracts and use --stop-on-attention.'
          : '- Stop after the requested stage and report exact contract/status/event paths.',
      ].join('\n');
      const summary = [
        `${stageLabel} · ${sizeLabel}`,
        `${state.modelGroups.length} model group${state.modelGroups.length === 1 ? '' : 's'}`,
        `${state.judgeSets.length} judge set${state.judgeSets.length === 1 ? '' : 's'}`,
        `${state.modules.length} module${state.modules.length === 1 ? '' : 's'}`,
      ].join(' · ');
      return {cli: commandLines.join('\n'), prompt, summary};
    }

    function readRunBuilderState(root) {
      if (!root) return;
      const checkedValues = (name) => [...root.querySelectorAll(`[data-run-builder="${name}"]:checked`)]
        .map((input) => input.value)
        .filter(Boolean);
      const radioValue = (name, fallback) => {
        const selected = root.querySelector(`[data-run-builder="${name}"]:checked`);
        return selected ? selected.value : fallback;
      };
      runBuilderState.modelGroups = checkedValues('modelGroups');
      runBuilderState.judgeSets = checkedValues('judgeSets');
      runBuilderState.modules = checkedValues('modules');
      runBuilderState.stage = radioValue('stage', runBuilderState.stage);
      runBuilderState.size = radioValue('size', runBuilderState.size);
      runBuilderState.runIdPrefix = root.querySelector('[data-run-builder="runIdPrefix"]')?.value || runBuilderState.runIdPrefix;
      runBuilderState.outputRoot = root.querySelector('[data-run-builder="outputRoot"]')?.value || runBuilderState.outputRoot;
      runBuilderState.maxActiveCalls = root.querySelector('[data-run-builder="maxActiveCalls"]')?.value || runBuilderState.maxActiveCalls;
    }

    function updateRunBuilderOutput() {
      const root = document.querySelector('.run-builder');
      const operator = (lastData || {}).operator || {};
      if (!root || !operator) return;
      const output = buildRunBuilderOutput(operator);
      const cli = root.querySelector('#runBuilderCli');
      const prompt = root.querySelector('#runBuilderPrompt');
      const summary = root.querySelector('#runBuilderSummary');
      const cliCopy = root.querySelector('[data-run-builder-copy="cli"]');
      const promptCopy = root.querySelector('[data-run-builder-copy="prompt"]');
      if (cli) cli.textContent = output.cli;
      if (prompt) prompt.textContent = output.prompt;
      if (summary) summary.textContent = output.summary;
      if (cliCopy) cliCopy.dataset.copy = output.cli;
      if (promptCopy) promptCopy.dataset.copy = output.prompt;
    }

    function handleRunBuilderAction(button) {
      const root = button.closest('.run-builder');
      if (!root) return;
      const action = button.dataset.runBuilderAction || '';
      const operator = (lastData || {}).operator || {};
      const defaults = runBuilderDefaults(operator);
      if (action === 'all-model-groups') {
        runBuilderState.modelGroups = defaults.groups;
      } else if (action === 'smoke-model-groups') {
        runBuilderState.modelGroups = defaults.groups.filter((name) => name.includes('smoke'));
        if (!runBuilderState.modelGroups.length && defaults.defaultGroup) {
          runBuilderState.modelGroups = [defaults.defaultGroup];
        }
      } else if (action === 'all-judges') {
        runBuilderState.judgeSets = defaults.judges;
      } else if (action === 'all-modules') {
        runBuilderState.modules = RUN_BUILDER_MODULES.map((module) => module.key);
      }
      openDetails.add('operator-run-builder');
      if (lastData) render(lastData);
      else updateRunBuilderOutput();
    }

    function triageCopy(module) {
      const state = module.score_state || {};
      const attention = module.attention || {};
      const statusPath = module.status_path || '';
      const eventsPath = module.events_path || '';
      return [
        '# Benchmark attention triage',
        '# Run this from the benchmark repo root: the directory containing suite_models.yaml.',
        'if [ ! -f suite_models.yaml ]; then echo "Run from the benchmark repo root." >&2; exit 2; fi',
        '',
        `# ${module.group || ''} / ${module.module_path || module.module || ''}`,
        `# Status: ${module.status || 'unknown'} / ${module.validity || 'unknown'}`,
        `# Score state: ${state.label || 'unknown'}`,
        `# Next action: ${attention.action || state.action || 'Inspect the run ledger directly.'}`,
        '# Do not score or promote this module until it is completed / score_ready.',
        '',
        statusPath ? `./venv/bin/python -m json.tool ${shellQuote(statusPath)} | sed -n '1,220p'` : '',
        eventsPath ? `tail -80 ${shellQuote(eventsPath)}` : '',
        '',
        '# Failure summary',
        failureCopy(module),
      ].filter(Boolean).join('\n');
    }

    function revealCopyText(value, copied) {
      copyPanel.hidden = false;
      copyPanelTitle.textContent = copied ? 'Copied to clipboard' : 'Copy manually';
      copyPanelNote.textContent = copied
        ? 'The text is also shown here in case this browser keeps its clipboard isolated.'
        : 'This browser blocked clipboard access. Select this text and copy it manually.';
      copyTextarea.value = value;
      copyTextarea.focus();
      copyTextarea.select();
    }

    async function copyValue(button, value) {
      const original = button.dataset.originalLabel || button.textContent;
      button.dataset.originalLabel = original;
      let copied = false;
      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(value);
          copied = true;
        } catch (error) {
          copied = false;
        }
      }
      revealCopyText(value, copied);
      button.textContent = copied ? 'Copied' : 'Select text';
      button.classList.toggle('copied', copied);
      window.setTimeout(() => {
        button.textContent = original;
        button.classList.remove('copied');
      }, 1400);
    }

    async function writeDisposition(button, statusPath, action) {
      const rejecting = action !== 'restore';
      const confirmedAction = rejecting ? 'reject' : 'restore';
      const pathParts = String(statusPath || '').replace(/\\/g, '/').split('/').filter(Boolean);
      const runId = pathParts.length >= 3 ? pathParts[pathParts.length - 3] : '';
      const confirmation = window.prompt(
        `Type "${confirmedAction} ${runId}" to confirm this disposition.`,
        '',
      );
      if (confirmation !== `${confirmedAction} ${runId}`) return;
      if (!csrfToken) {
        revealCopyText('Dashboard CSRF token is unavailable. Refresh this local page before retrying.', false);
        return;
      }
      const original = button.textContent;
      button.textContent = 'Saving...';
      button.disabled = true;
      try {
        const response = await fetch('/api/disposition', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Benchmark-CSRF': csrfToken,
          },
          body: JSON.stringify({
            status_path: statusPath,
            disposition: rejecting ? 'rejected_from_analysis' : 'candidate',
            reason: rejecting ? 'operator_rejected_malformed_or_incomplete_run' : 'operator_restored_diagnostic_run',
            notes: rejecting
              ? 'Excluded from scored analysis because the run is malformed, incomplete, or provider-failed.'
              : 'Restored to active dashboard consideration by operator.',
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        button.textContent = rejecting ? 'Rejected' : 'Restored';
	        const dataResponse = await fetch(detailUrl('/api/runs', {scope: evidenceRunScope}), {cache: 'no-cache'});
	        if (dataResponse.ok) {
	          runsEtag = dataResponse.headers.get('ETag') || '';
	          const data = await dataResponse.json();
	          carryForwardDetail(lastData, data);
	          await hydrateDashboardDetails(data);
	          render(data);
	        }
      } catch (error) {
        revealCopyText(String(error), false);
        button.textContent = 'Failed';
      } finally {
        window.setTimeout(() => {
          button.disabled = false;
          button.textContent = original;
        }, 1200);
      }
    }

    function rememberFresh(events, evidenceItems = [], evidenceLoaded = false) {
      const nextFresh = new Set();
      let freshEvidenceIndex = 0;
      for (const event of events || []) {
        const key = eventKey(event);
        if (!suppressFreshOnNextRender && !firstPaint && !seenEvents.has(key)) nextFresh.add(key);
        seenEvents.add(key);
      }
      for (const item of evidenceItems || []) {
        const key = evidenceKey(item);
        if (!suppressFreshOnNextRender && !firstEvidencePaint && !seenEvents.has(key)) {
          nextFresh.add(key);
          evidenceTraceBirths.set(
            key,
            performance.now() + Math.min(freshEvidenceIndex, 8) * EVIDENCE_TRACE_STAGGER_MS,
          );
          freshEvidenceIndex += 1;
        }
        seenEvents.add(key);
      }
      for (const key of evidenceTraceBirths.keys()) {
        if (!nextFresh.has(key)) evidenceTraceBirths.delete(key);
      }
      if (suppressFreshOnNextRender) evidenceTraceBirths.clear();
      if (evidenceLoaded) firstEvidencePaint = false;
      freshEvents = nextFresh;
      firstPaint = false;
      if (evidenceLoaded) suppressFreshOnNextRender = false;
    }

    function formatTime(value) {
      if (!value) return 'unknown';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString([], {month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit'});
    }

    function relativeTime(value) {
      if (!value) return 'No timestamp';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      const diff = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
      if (diff < 60) return `${diff}s ago`;
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return formatTime(value);
    }

    function duration(seconds) {
      if (seconds == null || Number.isNaN(Number(seconds))) return 'pending';
      const total = Math.max(0, Math.round(Number(seconds)));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = total % 60;
      if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
      if (minutes) return `${minutes}m ${String(secs).padStart(2, '0')}s`;
      return `${secs}s`;
    }

    function statusBadge(module) {
      return `<span class="badge ${esc(module.severity)}">${esc(module.status)} / ${esc(module.validity)}</span>`;
    }

    function failureClassificationMarkup(attention) {
      const classification = (attention || {}).classification || {};
      if (!classification.evidence_class) return '';
      const detail = [
        classification.provider,
        classification.provider_code,
        classification.retry_policy_kind,
        classification.action,
      ].filter(Boolean).join(' · ');
      return `
        <div class="failure-classification">
          <span class="badge">${esc(classification.label || `${classification.evidence_class} / ${classification.category || 'unclassified'}`)}</span>
          ${detail ? `<span class="stage-line">${esc(detail)}</span>` : ''}
        </div>
      `;
    }

    function metric(label, value, note, tone = '') {
      return `
        <article class="metric ${tone}">
          <div class="metric-label">${esc(label)}</div>
          <div class="metric-value">${esc(value)}</div>
          <div class="metric-note">${esc(note || '')}</div>
        </article>
      `;
    }

    function eventTitle(event) {
      const name = String(event.event || 'event').replaceAll('_', ' ');
      const target = [
        event.role,
        event.phase,
        event.model,
        event.scenario,
        event.test_type,
        event.item_idx != null ? `item ${event.item_idx}` : '',
        event.run_number != null ? `run ${event.run_number}` : '',
        event.side,
        event.turn ? `turn ${event.turn}` : '',
      ].filter(Boolean).join(' | ');
      return target ? `${name} - ${target}` : name;
    }

    function eventDetail(event) {
      return [
        event.group,
        event.module_path,
        event.stage,
        event.role,
        event.phase,
        event.failure_stage ? `failure stage: ${event.failure_stage}` : '',
        event.failure_reason,
      ].filter(Boolean).join(' | ');
    }

    function renderEvent(event) {
      const fresh = freshEvents.has(eventKey(event)) ? ' is-new' : '';
      return `
        <article class="event-row${fresh}">
          <div class="event-top">
            <div class="event-title">${esc(eventTitle(event))}</div>
            <span class="chip">${esc(relativeTime(event.timestamp))}</span>
          </div>
          <div class="event-detail">${esc(eventDetail(event))}</div>
        </article>
      `;
    }

    function allModules(data) {
      const modules = rawModules(data);
      return showAcknowledged ? modules : modules.filter((module) => !isHiddenModule(module));
    }

    function sortByActivity(a, b) {
      const rank = {running: 4, attention: 3, ready: 2, idle: 1};
      const rankDiff = (rank[b.severity] || 0) - (rank[a.severity] || 0);
      if (rankDiff) return rankDiff;
      return String(b.updated_at || '').localeCompare(String(a.updated_at || ''));
    }

    function turnCountLabel(progress) {
      const saved = Number(progress.turn_saved || 0);
      const planned = Number(progress.planned_turns || 0);
      if (planned > 0 && planned >= saved) return `${saved} / ${planned}`;
      if (saved > 0) return `${saved} saved`;
      if (planned > 0) return `0 / ${planned}`;
      return '0 / unknown';
    }

	    function conversationCountLabel(progress) {
	      const completed = Number(progress.conversations_completed || 0);
	      const started = Number(progress.conversations_started || 0);
	      if (started > 0 && started >= completed) return `${completed} / ${started}`;
	      if (completed > 0) return `${completed} complete`;
	      if (started > 0) return `0 / ${started}`;
	      return '0 / 0';
	    }

		    function stageClass(value) {
		      return `stage-${String(value || 'event').toLowerCase().replace(/[^a-z0-9_]+/g, '_')}`;
		    }

		    function queueAccentClass(item) {
		      const seed = [item?.run_id, item?.module, item?.module_path, item?.title].filter(Boolean).join('|');
		      let hash = 0;
		      for (let index = 0; index < seed.length; index += 1) {
		        hash = ((hash << 5) - hash + seed.charCodeAt(index)) | 0;
		      }
		      return `queue-accent-${Math.abs(hash) % 5}`;
		    }

	    function canonicalStage(value) {
	      const text = String(value || 'event').toLowerCase().replace(/[^a-z0-9_]+/g, '_');
	      const aliases = {
	        generation: 'generating',
	        run: 'generating',
	        scoring: 'judging',
	        ready: 'score_ready',
	        score_ready: 'score_ready',
	        failed: 'attention',
	        rejected_from_analysis: 'rejected',
	      };
	      return aliases[text] || text;
	    }

	    function stageLabel(value) {
	      const text = canonicalStage(value);
	      const labels = {
	        prepared: 'Prepared',
	        queued: 'Queued',
	        generating: 'Generation',
	        generation: 'Generation',
	        needs_scoring: 'Needs scoring',
	        judging: 'Judging',
	        scoring: 'Judging',
	        score_ready: 'Results',
	        ready: 'Results',
	        attention: 'Attention',
	        rejected: 'Rejected',
	        event: 'Event',
	      };
	      return labels[text] || text.replaceAll('_', ' ');
	    }

	    function visibleFlowLanes(data) {
	      return ((data.flow || {}).lanes || []).map((lane) => {
	        const scopedItems = (lane.items || []).filter((item) => runScopeMatches(item, data));
	        const items = showAcknowledged
	          ? scopedItems
	          : scopedItems.filter((item) => !isHiddenFlowItem(item));
	        return {
	          ...lane,
	          items,
	          count: items.length,
	          work_group_count: new Set(items.map((item) => `${item.title || ''}|${item.model_summary || ''}|${item.sample_summary || ''}`)).size,
	          complete_units: items.reduce((total, item) => total + Number(item.complete_units || 0), 0),
	          expected_units: items.reduce((total, item) => total + Number(item.expected_units || 0), 0),
	        };
	      });
	    }

		    function scopedModules(data) {
		      return workingModules(data);
		    }

		    function scopedFailedCount(data) {
		      return scopedModules(data).filter((module) => String(module.status || '').startsWith('failed')).length;
		    }

		    function scopedSpendIssueCount(data) {
		      return scopedModules(data).filter((module) => ((module.spend_guard || {}).severity === 'attention')).length;
		    }

		    function scopedCostTotal(data) {
		      if (evidenceRunScope === 'all') return Number((data.summary || {}).tracked_cost_total_usd || 0);
		      return scopedModules(data).reduce((total, module) => total + Number((module.cost || {}).total_cost_usd || 0), 0);
		    }

		    function scopedCostBreakdown(data) {
		      const modules = scopedModules(data);
		      return modules.reduce((totals, module) => {
		        const cost = module.cost || {};
		        totals.total += Number(cost.total_cost_usd || 0);
		        totals.reported += Number(cost.reported_cost_usd || 0);
		        totals.estimated += Number(cost.estimated_cost_usd || 0);
		        totals.unclassified += Number(cost.unclassified_cost_usd || 0);
		        return totals;
		      }, {total: 0, reported: 0, estimated: 0, unclassified: 0});
		    }

		    function scopedUnknownCostSummary(data) {
		      const rows = workingRawModules(data)
		        .map((module) => ({module, calls: Number((module.cost || {}).unknown_cost_calls || 0)}))
		        .filter((row) => row.calls > 0 && (row.module.spend_guard || {}).kind !== 'adapter_pricing_partial');
		      const models = new Set();
		      rows.forEach(({module}) => {
		        Object.keys((module.cost || {}).unknown_cost_by_model || {}).forEach((model) => models.add(model));
		      });
		      return {
		        calls: rows.reduce((total, row) => total + row.calls, 0),
		        modules: rows.length,
		        models: [...models].sort(),
		      };
		    }

		    function scopedAdapterCostSummary(data) {
		      const rows = workingRawModules(data)
		        .map((module) => ({module, calls: Number((module.cost || {}).unknown_cost_calls || 0)}))
		        .filter((row) => row.calls > 0 && (row.module.spend_guard || {}).kind === 'adapter_pricing_partial');
		      return {
		        calls: rows.reduce((total, row) => total + row.calls, 0),
		        modules: rows.length,
		      };
		    }

		    function currentLifecycleStage(data, queue) {
		      const stages = new Map(((queue || {}).stages || []).map((stage) => [stage.id, stage]));
		      const firstPresent = (ids) => ids.find((id) => Number((stages.get(id) || {}).count || 0));
		      const active = firstPresent(['attention', 'scoring', 'generating']);
		      const id = active || firstPresent(['queued', 'needs_scoring', 'prepared', 'score_ready']) || 'prepared';
		      const stage = stages.get(id) || {};
		      const notes = {
		        prepared: 'Contracts are ready; no paid calls have started.',
		        queued: 'Accepted by the scheduler and waiting for runner capacity.',
		        generating: 'Model responses are arriving and transcripts are being saved.',
		        needs_scoring: 'Generation is complete; awaiting scoring approval.',
		        scoring: 'Judge calls are active and result bundles are landing.',
		        score_ready: 'Scored artifacts are complete and ready for review.',
		        attention: 'A run-integrity issue needs review before more spend.',
		      };
		      return {
		        id,
		        label: stage.title || stageLabel(id),
		        note: id === 'attention'
		          ? ((stage.items || [])[0]?.next_action || notes[id])
		          : (notes[id] || (stage.items || [])[0]?.next_action || stage.description || ''),
		      };
		    }

		    function overallVerdict(data, visibleAttention, queue) {
		      const summary = data.summary || {};
		      const running = Number((queue || {}).active_units || 0) || scopedModules(data).filter((module) => module.severity === 'running').length;
		      const activeLeases = Number(summary.paid_call_active_count || 0);
	      const rateLimits = evidenceRunScope === 'all' ? Number(summary.paid_call_rate_limit_cooldown_count || 0) : 0;
	      if (visibleAttention || rateLimits) {
	        const issueCount = visibleAttention || rateLimits;
	        return {tone: 'attention', title: 'Stop / inspect', note: `${issueCount} run-integrity issue${issueCount === 1 ? '' : 's'} before spending`};
	      }
	      if (running || (evidenceRunScope === 'all' && activeLeases)) {
	        const activeModules = running || 0;
	        return {tone: 'running', title: 'Live run', note: `${activeModules} module${activeModules === 1 ? '' : 's'} active · global leases ${activeLeases}`};
	      }
	      const lifecycle = currentLifecycleStage(data, queue);
	      const tone = lifecycle.id === 'needs_scoring' ? 'review' : (lifecycle.id === 'queued' ? 'queued' : 'ready');
	      return {tone, title: lifecycle.label, note: lifecycle.note};
	    }

	    function hudMetric(label, value, note = '', className = '') {
	      return `<div class="hud-metric ${esc(className)}"><span>${esc(label)}</span><strong>${esc(value)}</strong><span class="hud-note">${esc(note)}</span></div>`;
	    }

	    function creditAlertState(module) {
	      const status = String(module.status || '').toLowerCase();
	      const attentionTitle = String((module.attention || {}).title || '').toLowerCase();
	      const scoreLabel = String((module.score_state || {}).label || '').toLowerCase();
	      const spendGuard = module.spend_guard || {};
	      if (
	        status === 'failed_billing'
	        || attentionTitle.includes('credits exhausted')
	        || scoreLabel === 'credits exhausted'
	      ) {
	        return 'exhausted';
	      }
	      if (spendGuard.severity === 'attention' && spendGuard.label === 'unpriced calls') {
	        return 'unpriced';
	      }
	      if (spendGuard.severity === 'attention' && spendGuard.label === 'low credit') {
	        return 'low';
	      }
	      return '';
	    }

	    function renderCreditAlert(data) {
	      const affected = workingRawModules(data)
	        .map((module) => ({module, state: creditAlertState(module)}))
	        .filter((item) => item.state);
	      if (!affected.length) return '';

	      const exhausted = affected.filter((item) => item.state === 'exhausted');
	      const unpriced = affected.filter((item) => item.state === 'unpriced');
	      const unknown = scopedUnknownCostSummary(data);
	      const selected = exhausted[0] || unpriced[0] || affected[0];
	      const module = selected.module;
	      const runLabel = `${module.run_id || module.group || 'run'} / ${module.module_path || module.module || 'module'}`;
	      const extraCount = affected.length - 1;
	      if (exhausted.length) {
	        const nextAction = (module.score_state || {}).action
	          || 'Refill the provider account, then resume the same prepared contract.';
	        return `
	          <section class="credit-alert attention" role="alert" aria-live="assertive">
	            <div class="credit-alert-label">Refill required</div>
	            <div class="credit-alert-copy">
	              <strong>Credits exhausted</strong>
	              <span>${esc(runLabel)}${extraCount ? ` and ${esc(extraCount)} more` : ''}. ${esc(nextAction)}</span>
	            </div>
	          </section>
	        `;
	      }
	      if (unpriced.length) {
	        const modelNote = unknown.models.length ? ` Models: ${unknown.models.join(', ')}.` : '';
	        return `
	          <section class="credit-alert warning" role="alert" aria-live="assertive">
	            <div class="credit-alert-label">Spend provenance</div>
	            <div class="credit-alert-copy">
	              <strong>Spend tracking incomplete</strong>
	              <span>${esc(unknown.calls)} recorded paid call${unknown.calls === 1 ? '' : 's'} across ${esc(unknown.modules)} module${unknown.modules === 1 ? '' : 's'} lack price metadata.${esc(modelNote)} This affects spend accounting, not benchmark validity.</span>
	            </div>
	          </section>
	        `;
	      }

	      const rawCreditRemaining = (module.cost || {}).credit_remaining_usd;
	      const creditRemaining = Number(rawCreditRemaining);
	      const remaining = rawCreditRemaining != null && Number.isFinite(creditRemaining)
	        ? `$${creditRemaining.toFixed(2)} remaining. `
	        : '';
	      return `
	        <section class="credit-alert warning" role="status" aria-live="polite">
	          <div class="credit-alert-label">Refill soon</div>
	          <div class="credit-alert-copy">
	            <strong>Credits running low</strong>
	            <span>${esc(runLabel)}${extraCount ? ` and ${esc(extraCount)} more` : ''}. ${esc(remaining)}Pause before the next paid batch and refill the provider account.</span>
	          </div>
	        </section>
	      `;
	    }

		    function runTiming(data, queue) {
		      const summary = data.summary || {};
		      const active = Number(queue.active_units || 0) > 0
		        || scopedModules(data).some((module) => module.severity === 'running');
		      const hasLedgerProgress = Number(queue.generation_completed_units || 0) > 0;
		      if (active) {
		        return {
		          label: 'Elapsed',
		          value: summary.active_elapsed && summary.active_elapsed !== 'none'
		            ? summary.active_elapsed
		            : summary.latest_elapsed || 'starting',
		          note: 'active run',
		        };
		      }
		      if (!hasLedgerProgress) {
		        return {label: 'Run time', value: 'Not started', note: 'no paid ledger writes'};
		      }
		      return {
		        label: 'Run duration',
		        value: summary.latest_elapsed || 'unknown',
		        note: 'selected historical run',
		      };
		    }

		    function renderHudRail(data, visibleAttention) {
		      const summary = data.summary || {};
		      const queue = scopedOperationalQueue(data);
	      const modules = scopedModules(data);
	      const attentionStage = (queue.stages || []).find((stage) => stage.id === 'attention') || {};
	      const attentionUnits = Number(queue.attention_units || attentionStage.units || visibleAttention || 0);
	      const attentionBits = [];
	      if (Number(attentionStage.count || 0)) attentionBits.push(`${attentionStage.count} group${attentionStage.count === 1 ? '' : 's'}`);
	      const failedCount = evidenceRunScope === 'all' ? Number(summary.failed_count || 0) : scopedFailedCount(data);
	      if (failedCount) attentionBits.push(`${failedCount} failed`);
	      if (evidenceRunScope === 'all' && Number(summary.paid_call_rate_limit_cooldown_count || 0)) attentionBits.push(`${summary.paid_call_rate_limit_cooldown_count} cooldown${summary.paid_call_rate_limit_cooldown_count === 1 ? '' : 's'}`);
	      const attentionNote = attentionBits.join(' · ') || 'clear';
	      const activeLeases = Number(summary.paid_call_active_count || 0);
	      const maxLeases = Number(summary.paid_call_max_active || 0);
	      const capacityNote = `${activeLeases} active · ${capacitySourceLabel(data)}`;
	      const costBreakdown = scopedCostBreakdown(data);
	      const adapterPricing = scopedAdapterCostSummary(data);
	      const costTotal = costBreakdown.total;
	      const costModules = evidenceRunScope === 'all' ? Number(data.module_count || 0) : modules.length;
	      const supplementalModules = modules.filter((module) => module.contract_membership === 'supplemental').length;
	      const costBasis = [];
	      if (costBreakdown.reported) costBasis.push(`${moneyCents(costBreakdown.reported)} provider-reported`);
	      if (costBreakdown.estimated) costBasis.push(`${moneyCents(costBreakdown.estimated)} estimated`);
	      if (costBreakdown.unclassified) costBasis.push(`${moneyCents(costBreakdown.unclassified)} basis unavailable`);
	      if (adapterPricing.calls) costBasis.push(`${adapterPricing.calls} adapter calls not individually priced`);
	      const ledgerNote = `${costModules} ${evidenceRunScope === 'all' ? 'loaded' : 'selected'} ledger${costModules === 1 ? '' : 's'}${supplementalModules ? `, ${supplementalModules} supplemental` : ''}`;
	      const costNote = [...costBasis, ledgerNote].join(' · ');
	      const spendLabel = adapterPricing.calls
	        ? 'Tracked spend to date (partial)'
	        : costBreakdown.unclassified || (costBreakdown.reported && costBreakdown.estimated)
	        ? 'Tracked spend to date'
	        : costBreakdown.reported
	          ? 'Provider-reported spend'
	          : 'Estimated spend to date';
	      const timing = runTiming(data, queue);
	      const generationProgress = progressFractionLabel(queue.generation_completed_units, queue.generation_expected_units);
	      const scoringProgress = progressFractionLabel(queue.score_completed_units, queue.score_expected_units);
	      const judgeProgress = progressFractionLabel(queue.judge_calls_completed, queue.judge_calls_expected);
		      const verdict = overallVerdict(data, visibleAttention, queue);
	      return `
	        <section class="hud-rail" aria-label="Run control status">
	          <div class="hud-verdict ${esc(verdict.tone)}">
	            <div class="hud-status">${esc(verdict.title)}</div>
	            <div class="hud-note">${esc(verdict.note)}</div>
	          </div>
	          ${hudMetric('Generation', generationProgress, 'transcripts')}
	          ${hudMetric('Scoring', scoringProgress, 'result bundles')}
	          ${hudMetric('Judge calls', judgeProgress, 'completed / planned')}
	          ${hudMetric('Capacity', `${activeLeases} / ${maxLeases}`, capacityNote, 'hud-capacity')}
	          ${hudMetric(timing.label, timing.value, timing.note)}
	          ${hudMetric('Latest write', relativeTime(summary.latest_event_at), formatTime(summary.latest_event_at))}
	          ${hudMetric(spendLabel, moneyCents(costTotal), costNote, 'hud-spend')}
	          ${hudMetric('Run integrity', attentionUnits ? `${attentionUnits} issues` : '0', attentionNote)}
	        </section>
	      `;
	    }

			    function updateTopSummary(data, visibleAttention) {
			      const summary = data.summary || {};
			      const queue = scopedOperationalQueue(data);
			      const timing = runTiming(data, queue);
			      const lanes = visibleFlowLanes(data);
			      const laneWorkGroups = Object.fromEntries(lanes.map((lane) => [lane.id, Number(lane.work_group_count || lane.count || 0)]));
			      const visibleRunning = workingModules(data).filter((module) => module.severity === 'running').length;
			      const active = Number(queue.active_units || 0) || Number(laneWorkGroups.generating || 0) + Number(laneWorkGroups.scoring || 0);
		      const running = Boolean(active || visibleRunning);
	      if (topComplete) topComplete.textContent = currentLifecycleStage(data, queue).label;
	      if (topElapsed) topElapsed.textContent = timing.value;
	      if (topElapsedLabel) topElapsedLabel.textContent = timing.label;
	      if (topActive) topActive.textContent = String(active || visibleRunning || 0);
	      if (topErrors) topErrors.textContent = String(visibleAttention || 0);
	      if (topErrorsStat) topErrorsStat.classList.toggle('attention', Boolean(visibleAttention));
	      if (topActiveStat) topActiveStat.classList.toggle('running', running);
	      if (brandShield) {
	        const nextSrc = running ? brandShield.dataset.runningSrc : brandShield.dataset.staticSrc;
	        if (nextSrc && brandShield.getAttribute('src') !== nextSrc) {
	          brandShield.setAttribute('src', nextSrc);
	        }
	        brandShield.classList.toggle('is-running', running);
	      }
	    }

	    function operationalQueue(data) {
	      return (data || {}).operational_queue || {stages: [], generation_expected_units: 0, generation_completed_units: 0, score_expected_units: 0, score_completed_units: 0, judge_calls_expected: 0, judge_calls_completed: 0, active_units: 0, attention_units: 0, leases: {}};
	    }

	    function queueWorkGroupCount(items) {
	      return new Set((items || []).map((item) => `${item.title || item.module || item.run_id || ''}|${item.model_summary || ''}|${item.sample_summary || ''}`)).size;
	    }

	    function scopedOperationalQueue(data) {
	      const queue = operationalQueue(data);
	      const stages = (queue.stages || []).map((stage) => {
	        const items = (stage.items || []).filter((item) => runScopeMatches(item, data));
	        const expectedUnits = items.reduce((total, item) => total + Number(item.expected_units || 0), 0);
	        const completedUnits = items.reduce((total, item) => total + Number(item.completed_units || 0), 0);
	        const activeUnits = items.reduce((total, item) => total + Number(item.active_units || 0), 0);
	        const attentionUnits = items.reduce((total, item) => total + Number(item.attention_units || 0), 0);
	        const units = items.reduce((total, item) => total + Number(item.units || 0), 0);
	        const generationExpectedUnits = items.reduce((total, item) => total + Number(item.generation_expected_units || 0), 0);
	        const generationCompletedUnits = items.reduce((total, item) => total + Number(item.generated_units || 0), 0);
	        const scoreExpectedUnits = items.reduce((total, item) => total + Number(item.score_expected_units || item.expected_score_units || 0), 0);
	        const scoreCompletedUnits = items.reduce((total, item) => total + Number(item.score_completed_units || 0), 0);
	        const judgeCallsExpected = items.reduce((total, item) => total + Number(item.judge_calls_expected || 0), 0);
	        const judgeCallsCompleted = items.reduce((total, item) => total + Number(item.judge_calls_completed || 0), 0);
	        const workGroups = queueWorkGroupCount(items);
	        return {
	          ...stage,
	          items,
	          count: items.length,
	          group_count: workGroups,
	          work_group_count: workGroups,
	          expected_units: expectedUnits,
	          completed_units: completedUnits,
	          active_units: activeUnits,
	          attention_units: attentionUnits,
	          units,
	          generation_expected_units: generationExpectedUnits,
	          generation_completed_units: generationCompletedUnits,
	          score_expected_units: scoreExpectedUnits,
	          score_completed_units: scoreCompletedUnits,
	          judge_calls_expected: judgeCallsExpected,
	          judge_calls_completed: judgeCallsCompleted,
	        };
	      });
	      return {
	        ...queue,
	        stages,
	        total_units: stages.reduce((total, stage) => total + Number(stage.expected_units || 0), 0),
	        generated_units: stages.reduce((total, stage) => total + Number(stage.completed_units || 0), 0),
	        active_units: stages.reduce((total, stage) => total + Number(stage.active_units || 0), 0),
	        attention_units: stages.reduce((total, stage) => total + Number(stage.attention_units || 0), 0),
	        generation_expected_units: stages.reduce((total, stage) => total + Number(stage.generation_expected_units || 0), 0),
	        generation_completed_units: stages.reduce((total, stage) => total + Number(stage.generation_completed_units || 0), 0),
	        score_expected_units: stages.reduce((total, stage) => total + Number(stage.score_expected_units || 0), 0),
	        score_completed_units: stages.reduce((total, stage) => total + Number(stage.score_completed_units || 0), 0),
	        judge_calls_expected: stages.reduce((total, stage) => total + Number(stage.judge_calls_expected || 0), 0),
	        judge_calls_completed: stages.reduce((total, stage) => total + Number(stage.judge_calls_completed || 0), 0),
	      };
	    }

		    function pct(numerator, denominator) {
		      const den = Number(denominator || 0);
		      if (!den) return 0;
		      return Math.max(0, Math.min(100, (Number(numerator || 0) / den) * 100));
		    }

		    function compactUnitLabel(value) {
		      const count = Number(value || 0);
		      return `${count} unit${count === 1 ? '' : 's'}`;
		    }

		    function progressFractionLabel(completed, expected) {
		      const total = Number(expected || 0);
		      if (!total) return compactUnitLabel(completed);
		      return `${Number(completed || 0)} / ${total}`;
		    }

		    function queueProgress(item, kind) {
		      if (kind === 'generation') {
		        return progressFractionLabel(item.generated_units ?? item.generation_completed_units, item.generation_expected_units);
		      }
		      if (kind === 'judges') {
		        return progressFractionLabel(item.judge_calls_completed, item.judge_calls_expected);
		      }
		      return progressFractionLabel(item.score_completed_units, item.score_expected_units ?? item.expected_score_units);
		    }

		    function queueNodeOpen(key, defaultOpen = false) {
		      return queueExpansionState.has(key) ? queueExpansionState.get(key) === true : defaultOpen;
		    }

		    function queueStageCount(stage) {
		      if (stage.id === 'prepared') return `${Number(stage.count || 0)} ready`;
		      if (stage.id === 'queued') return `${Number(stage.count || 0)} queued`;
		      if (stage.id === 'generating') return queueProgress(stage, 'generation');
		      if (['needs_scoring', 'scoring', 'score_ready'].includes(stage.id)) return queueProgress(stage, 'scoring');
		      if (stage.id === 'attention') return `${Number(stage.attention_units || stage.units || 0)} issues`;
		      return String(stage.count || 0);
		    }

		    function queueStageMeta(stage) {
		      const bits = [];
		      if (stage.id === 'prepared') bits.push('Contracts and commands ready; no paid calls started');
		      if (stage.id === 'queued') bits.push('Accepted by scheduler; waiting for runner capacity');
		      if (stage.id === 'generating') bits.push(`${queueProgress(stage, 'generation')} transcripts saved`);
		      if (stage.id === 'needs_scoring') {
		        bits.push(`${queueProgress(stage, 'generation')} transcripts complete`);
		        bits.push(`${queueProgress(stage, 'scoring')} result bundles`);
		        bits.push('awaiting scoring approval');
		      }
		      if (stage.id === 'scoring') {
		        bits.push(`${queueProgress(stage, 'scoring')} result bundles`);
		        bits.push(`${queueProgress(stage, 'judges')} judge calls`);
		      }
		      if (stage.id === 'score_ready') {
		        bits.push(`${queueProgress(stage, 'scoring')} result bundles ready`);
		        bits.push(`${queueProgress(stage, 'judges')} judge calls`);
		        if (Number(stage.excluded_score_units || 0)) bits.push(`${Number(stage.excluded_score_units)} excluded`);
		      }
		      if (Number(stage.active_units || 0)) bits.push(`${stage.active_units} active`);
		      if (Number(stage.count || 0)) bits.push(`${stage.count} module${stage.count === 1 ? '' : 's'}`);
		      return bits.join(' · ') || stage.description || '';
		    }

		    function queueItemCount(stage, item) {
		      if (stage.id === 'prepared') return 'ready';
		      if (stage.id === 'queued') return 'queued';
		      if (stage.id === 'generating') return queueProgress(item, 'generation');
		      if (['needs_scoring', 'scoring', 'score_ready'].includes(stage.id)) {
		        const label = item.score_unit_label || 'results';
		        return `${queueProgress(item, 'scoring')} ${label}`;
		      }
		      if (stage.id === 'attention') return `${Number(item.attention_units || item.units || 0)} issues`;
		      return item.units || 0;
		    }

		    function queueItemLifecycleMeta(stage, item) {
		      const generationLabel = item.generation_unit_label || 'transcripts';
		      const scoreLabel = item.score_unit_label || 'result bundles';
		      const bits = [];
		      if (stage.id === 'prepared') bits.push('contract ready', 'no paid calls');
		      if (stage.id === 'queued') bits.push('scheduler accepted', 'waiting for capacity');
		      if (['generating', 'needs_scoring', 'scoring', 'score_ready'].includes(stage.id)) {
		        bits.push(`${queueProgress(item, 'generation')} ${generationLabel}`);
		      }
		      if (['needs_scoring', 'scoring', 'score_ready'].includes(stage.id)) {
		        bits.push(`${queueProgress(item, 'scoring')} ${scoreLabel}`);
		        bits.push(`${queueProgress(item, 'judges')} judge calls`);
		      }
		      if (stage.id === 'needs_scoring') bits.push('awaiting scoring approval');
		      if (Number(item.active_units || 0)) bits.push(`${item.active_units} active`);
		      return bits;
		    }

		    function renderQueueModels(stage, item, groupKey) {
		      const models = item.models || [];
		      if (!models.length) return '';
		      return `
		        <div class="queue-model-list">
		          ${models.slice(0, 24).map((model) => {
		            const scoreStage = ['needs_scoring', 'scoring', 'score_ready'].includes(stage.id);
		            const expected = scoreStage
		              ? Number(model.expected_score_units || 0)
		              : Number(model.expected_units || 0);
		            const completed = scoreStage
		              ? Number(model.scored_units || 0)
		              : Number(model.completed_units || 0);
		            const fill = expected ? pct(completed, expected) : pct(model.active_units || completed, Math.max(1, item.expected_units || stage.expected_units || 1));
		            const progressKind = scoreStage ? 'scored' : 'generated';
		            const meta = [
		              scoreStage && Number(model.expected_units || 0) ? `${progressFractionLabel(model.completed_units, model.expected_units)} generated` : '',
		              expected ? `${progressFractionLabel(completed, expected)} ${progressKind}` : '',
		              Number(model.active_units || 0) ? `${model.active_units} active` : '',
		              model.model_id && model.model_id !== model.label ? model.model_id : '',
		            ].filter(Boolean).join(' · ');
		            return `
		              <div class="queue-model-row ${stageClass(stage.id)}">
		                <div class="queue-model-main">
			                  <div class="queue-model-title" title="${esc(model.label || model.id || 'model')}">${esc(model.label || model.id || 'model')}</div>
		                  <div class="queue-model-meta">${esc(meta || 'waiting for ledger events')}</div>
		                </div>
		                <div class="queue-model-count">${esc(expected ? progressFractionLabel(completed, expected) : model.active_units || completed || 0)}</div>
		                <div class="queue-fill" aria-hidden="true"><span style="--fill:${fill}%"></span></div>
		              </div>
		            `;
		          }).join('')}
		        </div>
		      `;
		    }

		    function renderQueueGroups(stage) {
		      const items = stage.items || [];
		      if (!items.length) return '';
		      return `
		        <div class="queue-group-list">
		          ${items.map((item, index) => {
		            const groupKey = `queue:${stage.id}:${item.id || item.title || item.module || item.run_id}`;
		            const open = queueNodeOpen(groupKey, false);
		            const isNew = !seenQueueGroupKeys.has(groupKey);
		            seenQueueGroupKeys.add(groupKey);
		            const scoreStage = ['needs_scoring', 'scoring', 'score_ready'].includes(stage.id);
		            const expected = scoreStage
		              ? Number(item.score_expected_units || item.expected_score_units || 0)
		              : Number(item.generation_expected_units || item.expected_units || 0);
		            const completed = scoreStage
		              ? Number(item.score_completed_units || 0)
		              : Number(item.generated_units || item.completed_units || 0);
		            const fill = expected ? pct(completed, expected) : pct(item.units || item.active_units || 0, Math.max(1, stage.expected_units || stage.units || 1));
		            const active = ['generating', 'scoring', 'attention'].includes(stage.id) && Number(item.active_units || 0) > 0;
		            const eta = ['queued', 'generating', 'scoring'].includes(stage.id) && item.eta_seconds != null
		              ? `ETA ${duration(item.eta_seconds)}`
		              : '';
		            const meta = [
		              ...queueItemLifecycleMeta(stage, item),
		              eta,
		            ].filter(Boolean).join(' · ');
		            return `
		              <div class="queue-group-row ${stageClass(stage.id)} ${queueAccentClass(item)} ${active ? 'is-active' : ''} ${isNew ? 'is-new-work' : ''}" style="--queue-order:${Math.min(index, 8)}">
		                <button class="queue-expander" type="button" data-queue-toggle="${esc(groupKey)}" aria-expanded="${open}" aria-label="${open ? 'Collapse' : 'Expand'} ${esc(item.title || item.module || 'work group')}">›</button>
		                <div class="queue-group-main">
			                  <div class="queue-group-title" title="${esc(item.title || item.module || item.run_id || 'work group')}">${esc(item.title || item.module || item.run_id || 'work group')}</div>
		                  <div class="queue-group-meta">${esc(meta || item.next_action || '')}</div>
		                </div>
		                <div class="queue-group-count">${esc(queueItemCount(stage, item))}</div>
		                <div class="queue-fill" aria-hidden="true"><span style="--fill:${fill}%"></span></div>
		              </div>
		              ${open ? renderQueueModels(stage, item, groupKey) : ''}
		            `;
		          }).join('')}
		        </div>
		      `;
		    }

		    function renderOperationalStage(stage, queue, changed) {
		      const key = `queue:${stage.id}`;
		      const hasItems = Boolean((stage.items || []).length);
		      const defaultOpen = hasItems && (
		        stage.id === 'needs_scoring'
		        || (stage.id === 'attention' && Number(stage.units || 0) > 0)
		        || (['generating', 'scoring'].includes(stage.id) && Number(stage.active_units || 0) > 0)
		      );
		      const open = queueNodeOpen(key, defaultOpen);
		      const total = Math.max(1, Number(queue.generation_expected_units || stage.expected_units || stage.units || 1));
		      const fill = stage.id === 'generating'
		        ? pct(stage.generation_completed_units, stage.generation_expected_units)
		        : ['needs_scoring', 'scoring', 'score_ready'].includes(stage.id)
		          ? pct(stage.score_completed_units, stage.score_expected_units)
		          : pct(stage.units, total);
		      const active = ['generating', 'scoring', 'attention'].includes(stage.id) && Number(stage.active_units || 0) > 0;
		      const filterStage = canonicalStage(stage.id);
		      return `
		        <div class="queue-stage-block">
		          <div class="queue-row ${stageClass(stage.id)} ${active ? 'is-active' : ''} ${changed ? 'changed' : ''} ${activeStageFilter === filterStage ? 'is-filtered' : ''}">
		            <button class="queue-expander" type="button" data-queue-toggle="${esc(key)}" aria-expanded="${open}" aria-label="${open ? 'Collapse' : 'Expand'} ${esc(stage.title)}">›</button>
		            <button class="queue-row-main" type="button" data-stage-filter="${esc(filterStage)}" aria-pressed="${activeStageFilter === filterStage}" title="${esc(stage.description || '')}">
		              <div class="queue-row-title">${esc(stage.title || stageLabel(stage.id))}</div>
		              <div class="queue-row-meta">${esc(queueStageMeta(stage))}</div>
		            </button>
		            <div class="queue-row-count">${esc(queueStageCount(stage))}</div>
		            <div class="queue-fill" aria-hidden="true"><span style="--fill:${fill}%"></span></div>
		          </div>
		          ${open ? renderQueueGroups(stage) : ''}
		        </div>
		      `;
		    }

			    function renderWorkQueue(data) {
			      const queue = scopedOperationalQueue(data);
		      const stages = queue.stages || [];
		      const lifecycle = currentLifecycleStage(data, queue);
		      const queueContext = `${lifecycle.label}: ${lifecycle.note}`;
		      const nextSignatures = new Map();
		      const rows = stages.map((stage) => {
		        const signature = [stage.units, stage.generation_completed_units, stage.score_completed_units, stage.judge_calls_completed, stage.active_units, stage.attention_units].join('/');
		        const changed = previousQueueStageSignatures.has(stage.id) && previousQueueStageSignatures.get(stage.id) !== signature;
		        nextSignatures.set(stage.id, signature);
		        return renderOperationalStage(stage, queue, changed);
		      }).join('');
		      previousQueueStageSignatures.clear();
		      nextSignatures.forEach((value, key) => previousQueueStageSignatures.set(key, value));
		      return `
		        <section class="panel queue-panel">
		          <div class="panel-head">
		            <div>
		              <h2>Work queue</h2>
		              <div class="panel-kicker">${esc(queueContext)}</div>
		            </div>
		            <button class="copy-button" type="button" data-stage-filter="all" aria-pressed="${activeStageFilter === 'all'}">All stages</button>
		          </div>
		          <div class="queue-list">${rows || '<div class="queue-detail-empty">No operational queue data yet.</div>'}</div>
		        </section>
		      `;
		    }

	    function evidenceKey(item) {
	      return [item.kind, item.group, item.module_path, item.event, item.timestamp, item.turn, item.judge_model, item.dimension, item.transcript_path, item.score_path].join('|');
	    }

	    function renderCheck(label, ok) {
	      return `<span class="check ${ok ? 'pass' : 'fail'}">${esc(label)}</span>`;
	    }

	    function evidenceItemStage(item) {
	      return canonicalStage(item?.problem ? 'attention' : (item?.stage || 'event'));
	    }

	    function filteredEvidenceItems(items) {
	      if (activeStageFilter === 'all') return items;
	      return (items || []).filter((item) => {
	        const stage = evidenceItemStage(item);
	        return stage === activeStageFilter;
	      });
	    }

	    function evidenceEntryKey(item, index) {
	      return `${evidenceKey(item)}|${index}`;
	    }

	    function evidenceEntries(items) {
	      return (items || []).map((item, index) => ({
	        item,
	        index,
	        key: evidenceEntryKey(item, index),
	      }));
	    }

	    function latestRunId(data = lastData) {
	      return (data?.summary || {}).latest_run_id || (data?.groups || [])[0]?.run_id || '';
	    }

	    function familyForScope(scope = evidenceRunScope, data = lastData) {
	      if (!String(scope || '').startsWith('family:')) return null;
	      const sha = String(scope).slice('family:'.length);
	      return ((data || {}).families || []).find((family) => family.prereg_sha256 === sha) || null;
	    }

	    function familyMemberIds(scope = evidenceRunScope, data = lastData) {
	      const family = familyForScope(scope, data);
	      return family ? (family.member_run_ids || []).map(String) : [];
	    }

	    function workflowForScope(scope = evidenceRunScope, data = lastData) {
	      if (!String(scope || '').startsWith('workflow:')) return null;
	      return ((data || {}).workflows || []).find((workflow) => workflow.key === scope) || null;
	    }

	    function scopeMemberIds(scope = evidenceRunScope, data = lastData) {
	      const familyIds = familyMemberIds(scope, data);
	      if (familyIds.length) return familyIds;
	      const workflow = workflowForScope(scope, data);
	      return workflow ? (workflow.member_run_ids || []).map(String) : [];
	    }

	    function runScopeLabel(data = lastData) {
	      if (evidenceRunScope === 'all') return 'All runs';
	      if (evidenceRunScope === 'latest') return 'Latest run';
	      const workflow = workflowForScope(evidenceRunScope, data);
	      if (workflow) return `Current workflow · ${workflow.member_count} runs`;
	      const family = familyForScope(evidenceRunScope, data);
	      if (family) return `Family · ${shortHash(family.prereg_sha256)} (${family.member_count} runs)`;
	      return shortHash(evidenceRunScope);
	    }

	    function runScopeMatches(item, data = lastData) {
	      if (evidenceRunScope === 'all') return true;
	      const members = scopeMemberIds(evidenceRunScope, data);
	      if (members.length) return members.includes(String(item.group || item.run_id || ''));
	      const runId = evidenceRunScope === 'latest' ? latestRunId(data) : evidenceRunScope;
	      return !runId || String(item.group || item.run_id || '') === String(runId);
	    }

	    function resolvedRunScope(data = lastData) {
	      if (evidenceRunScope === 'all') return 'all';
	      if (String(evidenceRunScope).startsWith('family:')) return evidenceRunScope;
	      return evidenceRunScope === 'latest' ? latestRunId(data) : evidenceRunScope;
	    }

	    function detailUrl(path, params) {
	      const query = new URLSearchParams(params);
	      return `${path}?${query.toString()}`;
	    }

	    async function fetchDashboardDetail(url) {
	      const headers = detailEtags.has(url) ? {'If-None-Match': detailEtags.get(url)} : {};
	      const response = await fetch(url, {cache: 'no-cache', headers});
	      if (response.status === 304) return detailCache.get(url);
	      if (!response.ok) throw new Error(`HTTP ${response.status}`);
	      const payload = await response.json();
	      const etag = response.headers.get('ETag');
	      if (etag) detailEtags.set(url, etag);
	      detailCache.set(url, payload);
	      return payload;
	    }

	    function evidenceDetailUrl(data = lastData) {
	      return detailUrl('/api/evidence', {
	        scope: evidenceRunScope,
	        stage: activeStageFilter,
	        content: evidenceContentFilter,
	        window: evidenceTraceWindow,
	      });
	    }

	    function carryForwardDetail(prev, next) {
	      // Carry the previous poll's evidence/contract detail onto the fresh
	      // snapshot when the run selection is unchanged, so the panels keep
	      // showing prior results instead of blanking while the new detail
	      // request is in flight.
	      if (!prev || !next) return;
	      if (evidenceRunFingerprint(prev) !== evidenceRunFingerprint(next)) return;
	      if (!next.evidence_feed && Array.isArray(prev.evidence_feed)) {
	        next.evidence_feed = prev.evidence_feed;
	        next.evidence_total_count = prev.evidence_total_count;
	      }
	      if ((!next.contracts || !next.contracts.length) && Array.isArray(prev.contracts) && prev.contracts.length) {
	        next.contracts = prev.contracts;
	        next.contract_detail_summary = prev.contract_detail_summary;
	        next.contract_detail_scope = prev.contract_detail_scope;
	      }
	    }

	    async function ensureEvidenceDetails(
	      data = lastData,
	      {renderOnComplete = true, requireCurrent = true} = {},
	    ) {
	      if (!data) return;
	      const url = evidenceDetailUrl(data);
	      const requestKey = `${url}|${runsEtag || data.generated_at || ''}`;
	      if (data.evidence_detail_key === requestKey || evidenceRequestKey === requestKey) return;
	      evidenceRequestKey = requestKey;
	      const sequence = ++evidenceRequestSequence;
	      const isCurrent = () => !requireCurrent || lastData === data;
	      data.evidence_loading = true;
	      data.evidence_detail_error = '';
	      // Only paint a loading state when we have nothing carried to show.
	      if (renderOnComplete && isCurrent() && !(data.evidence_feed && data.evidence_feed.length)) render(data);
	      try {
	        const payload = await fetchDashboardDetail(url);
	        if (sequence !== evidenceRequestSequence || !isCurrent() || !payload) return;
	        data.evidence_feed = payload.items || [];
	        data.evidence_total_count = Number(payload.total_count || 0);
	        data.evidence_detail_key = requestKey;
	      } catch (error) {
	        if (sequence !== evidenceRequestSequence || !isCurrent()) return;
	        data.evidence_feed = [];
	        data.evidence_detail_key = requestKey;
	        data.evidence_detail_error = String(error);
	      } finally {
	        if (sequence === evidenceRequestSequence) evidenceRequestKey = '';
	        if (sequence === evidenceRequestSequence && isCurrent()) {
	          data.evidence_loading = false;
	          if (renderOnComplete) render(data);
	        }
	      }
	    }

	    function contractDetailUrl() {
	      return detailUrl('/api/contracts', {scope: evidenceRunScope});
	    }

	    async function ensureContractDetails(
	      data = lastData,
	      {renderOnComplete = true, requireCurrent = true} = {},
	    ) {
	      if (!data) return;
	      const url = contractDetailUrl();
	      const requestKey = `${url}|${runsEtag || data.generated_at || ''}`;
	      if (data.contract_detail_key === requestKey || contractRequestKey === requestKey) return;
	      contractRequestKey = requestKey;
	      const sequence = ++contractRequestSequence;
	      const isCurrent = () => !requireCurrent || lastData === data;
	      data.contract_loading = true;
	      data.contract_detail_error = '';
	      // Only paint a loading state when we have nothing carried to show.
	      if (renderOnComplete && isCurrent() && !(data.contracts && data.contracts.length)) render(data);
	      try {
	        const payload = await fetchDashboardDetail(url);
	        if (sequence !== contractRequestSequence || !isCurrent() || !payload) return;
	        data.contracts = payload.contracts || [];
	        data.contract_detail_summary = payload.summary || {};
	        data.contract_detail_scope = payload.resolved_scope || resolvedRunScope(data);
	        data.contract_detail_key = requestKey;
	      } catch (error) {
	        if (sequence !== contractRequestSequence || !isCurrent()) return;
	        data.contracts = [];
	        data.contract_detail_key = requestKey;
	        data.contract_detail_error = String(error);
	      } finally {
	        if (sequence === contractRequestSequence) contractRequestKey = '';
	        if (sequence === contractRequestSequence && isCurrent()) {
	          data.contract_loading = false;
	          if (renderOnComplete) render(data);
	        }
	      }
	    }

	    async function hydrateDashboardDetails(data) {
	      const options = {renderOnComplete: false, requireCurrent: false};
	      const requests = [ensureEvidenceDetails(data, options)];
	      if (openDetails.has('run-contract')) {
	        requests.push(ensureContractDetails(data, options));
	      }
	      await Promise.allSettled(requests);
	    }

	    function evidenceRunFingerprint(data = lastData) {
	      const runId = evidenceRunScope === 'latest' ? latestRunId(data) : evidenceRunScope;
	      return `${evidenceRunScope}:${runId || ''}`;
	    }

	    function resetEvidenceFiltersToAll() {
	      activeStageFilter = 'all';
	      evidenceContentFilter = 'all';
	      evidenceTraceWindow = DEFAULT_EVIDENCE_TRACE_WINDOW;
	      evidenceAutoFollow = true;
	      evidenceTraceAutoFollow = true;
	      selectedEvidenceKey = '';
	      lastEvidenceViewSignature = '';
	    }

	    function syncEvidenceDefaultsForRun(data) {
	      const fingerprint = evidenceRunFingerprint(data);
	      if (!fingerprint) return;
	      if (!lastEvidenceRunFingerprint) {
	        lastEvidenceRunFingerprint = fingerprint;
	        return;
	      }
	      if (fingerprint !== lastEvidenceRunFingerprint) {
	        resetEvidenceFiltersToAll();
	        pendingFeedPanelScroll = true;
	      }
	      lastEvidenceRunFingerprint = fingerprint;
	    }

	    function scopedLatestEvents(data) {
	      return (data?.latest_events || []).filter((event) => runScopeMatches(event, data));
	    }

	    function workingRawModules(data) {
	      return rawModules(data).filter((module) => runScopeMatches(module, data));
	    }

	    function workingModules(data) {
	      const modules = workingRawModules(data);
	      return showAcknowledged ? modules : modules.filter((module) => !isHiddenModule(module));
	    }

	    function filteredEvidenceEntries(items) {
	      const entries = evidenceEntries(items);
	      return entries.filter((entry) => {
	        if (!runScopeMatches(entry.item)) return false;
	        const stage = evidenceItemStage(entry.item);
	        if (activeStageFilter !== 'all' && stage !== activeStageFilter) return false;
	        if (evidenceContentFilter === 'all') return true;
	        if (evidenceContentFilter === 'writes') return entry.item.kind !== 'turn_pair';
	        return entry.item.kind === 'turn_pair' || stage === 'attention';
	      });
	    }

	    function windowedEvidenceEntries(entries) {
	      if (evidenceTraceWindow === 'all') return entries;
	      const windowSize = Number(evidenceTraceWindow);
	      if (!Number.isFinite(windowSize) || windowSize <= 0) return entries;
	      return entries.slice(-windowSize);
	    }

	    function evidenceWindowLabel() {
	      return evidenceTraceWindow === 'all' ? 'All-time' : `Last ${evidenceTraceWindow}`;
	    }

	    function evidenceContentLabel() {
	      const labels = {
	        text: 'Text',
	        all: 'All evidence',
	        writes: 'Writes',
	      };
	      return labels[evidenceContentFilter] || 'Text';
	    }

	    function evidenceItemSize(item) {
	      const user = item.user_message || '';
	      const answer = item.model_response || '';
	      const copy = item.problem || item.event || '';
	      const judgeResult = item.judge_result == null ? '' : JSON.stringify(item.judge_result);
	      return Math.max(8, String(user).length + String(answer).length + String(copy).length + judgeResult.length);
	    }

	    function evidenceItemTitle(item) {
	      if (item.kind === 'turn_pair') return `${item.module || 'module'} evidence`;
	      if (item.event === 'judge_result_parsed') {
	        const dimension = String(item.dimension || 'result').replaceAll('_', ' ');
	        return `${dimension} judged`;
	      }
	      return String(item.event || 'event').replaceAll('_', ' ');
	    }

	    function judgeResultCopy(item) {
	      if (item.event !== 'judge_result_parsed') return '';
	      const result = item.judge_result;
	      if (result && typeof result === 'object' && !Array.isArray(result)) {
	        return Object.entries(result)
	          .map(([key, value]) => `${String(key).replaceAll('_', ' ')}: ${String(value)}`)
	          .join(' · ');
	      }
	      const maximum = item.max_score != null ? ` / ${item.max_score}` : '';
	      return `Result: ${String(result)}${maximum}`;
	    }

	    function evidenceItemMeta(item) {
	      return [
	        item.group,
	        item.module_path,
	        item.evidence_class && item.category ? `${item.evidence_class} / ${item.category}` : '',
	        item.provider_code,
	        item.test_type,
	        item.item_idx != null ? `item ${item.item_idx}` : '',
	        item.judge_model ? `judge ${item.judge_model}` : '',
	        item.dimension ? String(item.dimension).replaceAll('_', ' ') : '',
	        item.side,
	        item.turn ? `turn ${item.turn}${item.planned_turns ? `/${item.planned_turns}` : ''}` : '',
	      ].filter(Boolean).join(' · ');
	    }

			    function evidenceTraceColor(stage) {
			      const colors = {
	        generating: '--color-accent',
	        judging: '--color-judging',
	        scoring: '--color-judging',
	        score_ready: '--color-good',
	        ready: '--color-good',
	        attention: '--color-bad',
		        queued: '--color-warn',
		        needs_scoring: '--color-review',
	        prepared: '--color-muted',
	        rejected: '--color-muted',
	        event: '--color-muted',
	      };
	      const variable = colors[canonicalStage(stage)] || '--color-muted';
	      const value = getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
		      return value || '#756f66';
			    }

			    const EVIDENCE_TRACE_MODULES = {
			      aita: {identity: 'aita', label: 'AITA', token: '--color-judging'},
			      aite: {identity: 'aita', label: 'AITA', token: '--color-judging'},
			      epis: {identity: 'epistemic', label: 'Epistemic', token: '--color-warn'},
			      epistemic: {identity: 'epistemic', label: 'Epistemic', token: '--color-warn'},
			      sus: {identity: 'sus', label: 'SUS', token: '--color-review'},
			    };
			    const EVIDENCE_TRACE_MARK_FRACTION = 0.82;

			    function evidenceTraceModel(item) {
			      const raw = String(
			        item?.model || item?.model_id || item?.target_model || item?.judge_model || '',
			      ).trim();
			      if (!raw) return null;
			      const record = modelRecord(raw);
			      const identity = String(record?.model_id || record?.key || raw).toLowerCase();
			      const label = modelDisplayParts(raw).name || titleCaseModel(raw) || raw;
			      return {raw, identity, label};
			    }

			    function evidenceTraceModule(item) {
			      const raw = String(item?.module_path || item?.module || '').trim();
			      if (!raw) return {identity: 'other', label: 'Other', token: '--color-muted'};
			      const key = raw.toLowerCase().replace(/[^a-z0-9]+/g, '_');
			      return EVIDENCE_TRACE_MODULES[key]
			        || {identity: key, label: raw, token: '--color-model-5'};
			    }

			    function evidenceTraceModuleColor(item) {
			      const module = evidenceTraceModule(item);
			      return canvasToken(module.token, '#287f78');
			    }

			    function evidenceTraceMotionReduced() {
			      return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
			    }

			    function evidenceTraceEnterProgress(item, now) {
			      const key = evidenceKey(item);
			      const startedAt = evidenceTraceBirths.get(key);
			      if (startedAt == null || evidenceTraceMotionReduced()) {
			        evidenceTraceBirths.delete(key);
			        return 1;
			      }
			      const linear = (now - startedAt) / EVIDENCE_TRACE_ENTER_MS;
			      if (linear >= 1) {
			        evidenceTraceBirths.delete(key);
			        return 1;
			      }
			      if (linear <= 0) return 0;
			      return 1 - Math.pow(1 - linear, 3);
			    }

			    function syncEvidenceTraceLegend(entries) {
			      const legend = document.getElementById('evidenceTraceLegend');
			      if (!legend) return;
			      const modules = new Map();
			      const models = new Map();
			      for (const entry of entries) {
			        const module = evidenceTraceModule(entry.item);
			        if (!modules.has(module.identity)) {
			          modules.set(module.identity, {...module, color: evidenceTraceModuleColor(entry.item)});
			        }
			        const model = evidenceTraceModel(entry.item);
			        if (model && !models.has(model.identity)) {
			          models.set(model.identity, model);
			        }
			      }
			      const visible = Array.from(modules.values()).slice(0, 5);
			      const signature = JSON.stringify({
			        modules: Array.from(modules.values()).map((module) => [module.identity, module.label, module.color]),
			        models: Array.from(models.values()).map((model) => [model.identity, model.label]),
			      });
			      if (legend.dataset.signature === signature) return;
			      legend.dataset.signature = signature;
			      const moduleItems = visible.map((module) => `
			        <span class="evidence-trace-legend-item" title="${esc(module.label)} evidence">
			          <span class="evidence-trace-legend-line" style="--legend-color:${esc(module.color)}" aria-hidden="true"></span>
			          <span>${esc(module.label)}</span>
			        </span>
			      `).join('');
			      const more = modules.size > visible.length
			        ? `<span class="evidence-trace-legend-more">+${modules.size - visible.length} modules</span>`
			        : '';
			      const modelSummary = models.size === 1
			        ? `<span class="evidence-trace-legend-more">Model · ${esc(Array.from(models.values())[0].label)}</span>`
			        : models.size > 1
			          ? `<span class="evidence-trace-legend-more">${models.size} models</span>`
			          : '<span class="evidence-trace-legend-more">Model not recorded</span>';
			      legend.innerHTML = `
			        <div class="evidence-trace-model-legend">${moduleItems}${more}${modelSummary}</div>
			      `;
			    }

	    function canvasToken(name, fallback) {
	      const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
	      return value || fallback;
	    }

	    function evidenceTraceWindowButton(value, label) {
	      const active = String(evidenceTraceWindow) === String(value);
	      return `<button class="stage-filter stage-filter-all" type="button" data-evidence-window="${esc(value)}" aria-pressed="${active}">${esc(label)}</button>`;
	    }

	    function evidenceContentButton(value, label) {
	      const active = String(evidenceContentFilter) === String(value);
	      return `<button class="stage-filter stage-filter-all" type="button" data-evidence-content="${esc(value)}" aria-pressed="${active}">${esc(label)}</button>`;
	    }

	    function runScopeOptions(data) {
	      const groups = data?.run_index || data?.groups || [];
	      const latest = latestRunId(data);
	      const options = [
	      ];
	      for (const workflow of (data?.workflows || [])) {
	        if (!workflow.key) continue;
	        options.push({
	          value: workflow.key,
	          label: `Current workflow · ${workflow.member_count || (workflow.member_run_ids || []).length} runs`,
	        });
	      }
	      options.push(
	        {value: 'latest', label: latest ? `Latest child · ${shortHash(latest)}` : 'Latest child run'},
	        {value: 'all', label: 'All runs'},
	      );
	      for (const family of (data?.families || [])) {
	        if (!family.prereg_sha256) continue;
	        options.push({
	          value: family.key || `family:${family.prereg_sha256}`,
	          label: `Family · ${shortHash(family.prereg_sha256)} (${family.member_count || (family.member_run_ids || []).length} runs)`,
	        });
	      }
	      for (const group of groups.slice(0, 18)) {
	        if (!group.run_id) continue;
	        options.push({
	          value: group.run_id,
	          label: `${shortHash(group.run_id)} · ${relativeTime(group.latest_event_at || group.updated_at)}`,
	        });
	      }
	      return options;
	    }

	    function evidenceRunSelect(data) {
	      const values = new Set(runScopeOptions(data).map((option) => option.value));
	      if (!values.has(evidenceRunScope)) evidenceRunScope = 'latest';
	      const options = runScopeOptions(data).map((option) => (
	        `<option value="${esc(option.value)}" ${option.value === evidenceRunScope ? 'selected' : ''}>${esc(option.label)}</option>`
	      )).join('');
	      return `<select class="stage-select" data-evidence-run-scope aria-label="Evidence run scope">${options}</select>`;
	    }

	    function syncTopScopeControl(data) {
	      if (!topScopeControl) return;
	      const nextOptions = runScopeOptions(data);
	      const existing = topScopeControl.querySelector('[data-evidence-run-scope]');
	      if (existing && document.activeElement === existing) return;
	      if (existing) {
	        const currentOptions = Array.from(existing.options || []);
	        const sameValues = currentOptions.length === nextOptions.length
	          && currentOptions.every((option, index) => option.value === nextOptions[index].value);
	        if (sameValues) {
	          currentOptions.forEach((option, index) => { option.textContent = nextOptions[index].label; });
	          existing.value = evidenceRunScope;
	          return;
	        }
	      }
	      topScopeControl.innerHTML = `
	        <span class="top-scope-label">Scope</span>
	        ${evidenceRunSelect(data).replace('aria-label="Evidence run scope"', 'aria-label="Benchmark run scope"')}
	      `;
	    }

	    function syncEvidenceWindowButtons() {
	      document.querySelectorAll('[data-evidence-window]').forEach((button) => {
	        button.setAttribute('aria-pressed', String(String(button.dataset.evidenceWindow) === String(evidenceTraceWindow)));
	      });
	    }

	    function syncEvidenceContentButtons() {
	      document.querySelectorAll('[data-evidence-content]').forEach((button) => {
	        button.setAttribute('aria-pressed', String(String(button.dataset.evidenceContent) === String(evidenceContentFilter)));
	      });
	    }

		    function traceEntriesForData(data) {
		      return windowedEvidenceEntries(filteredEvidenceEntries((data || {}).evidence_feed || []));
		    }

		    function sampledEvidenceTraceEntries(entries, width) {
		      const maxMarks = Math.max(25, Math.floor(width / 4));
		      if (entries.length <= maxMarks) return entries;
		      const indices = new Set([0, entries.length - 1]);
		      const selectedIndex = entries.findIndex((entry) => entry.key === selectedEvidenceKey);
		      if (selectedIndex >= 0) indices.add(selectedIndex);
		      const identity = (entry) => {
		        const model = evidenceTraceModel(entry.item);
		        const module = evidenceTraceModule(entry.item);
		        return `${model?.identity || ''}|${module.identity}|${evidenceItemStage(entry.item)}`;
		      };
		      for (let index = 1; index < entries.length; index += 1) {
		        if (identity(entries[index - 1]) !== identity(entries[index])) {
		          indices.add(index - 1);
		          indices.add(index);
		        }
		      }
		      for (let offset = 0; offset < maxMarks; offset += 1) {
		        indices.add(Math.round(offset * (entries.length - 1) / (maxMarks - 1)));
		      }
		      return Array.from(indices)
		        .sort((a, b) => a - b)
		        .map((index) => entries[index]);
		    }

	    function evidenceViewSignature(data) {
	      return JSON.stringify({
	        scope: evidenceRunScope,
	        stage: activeStageFilter,
	        content: evidenceContentFilter,
	        window: evidenceTraceWindow,
	        keys: traceEntriesForData(data).map((entry) => entry.key),
	      });
	    }

		    function resizeEvidenceTraceCanvas(canvas, context) {
		      const ratio = window.devicePixelRatio || 1;
		      const rect = canvas.getBoundingClientRect();
		      const width = Math.max(1, Math.floor(rect.width * ratio));
		      const height = Math.max(1, Math.floor(rect.height * ratio));
		      if (canvas.width !== width || canvas.height !== height) {
		        canvas.width = width;
		        canvas.height = height;
		      }
		      context.setTransform(ratio, 0, 0, ratio, 0, 0);
		    }

	    function evidenceTraceMode() {
	      return 'horizontal';
	    }

	    function sizeEvidenceTraceCanvas(canvas) {
	      const wrap = canvas.parentElement;
	      canvas.style.width = '100%';
	      canvas.style.height = '';
	      if (wrap) wrap.scrollLeft = 0;
	    }

	    function scrollEvidenceTraceToLive() {
	      const canvas = document.getElementById('evidenceTraceCanvas');
	      const wrap = canvas?.parentElement;
	      if (!wrap) return;
	      wrap.scrollLeft = Math.max(0, wrap.scrollWidth - wrap.clientWidth);
	    }

	    function requestEvidenceTraceLiveScroll() {
	      if (!evidenceTraceAutoFollow) return;
	      window.requestAnimationFrame(scrollEvidenceTraceToLive);
	    }

	    function syncEvidenceTraceAccessibility(canvas, entries) {
	      if (!entries.length) {
	        canvas.tabIndex = -1;
	        canvas.setAttribute('aria-valuemin', '0');
	        canvas.setAttribute('aria-valuemax', '0');
	        canvas.setAttribute('aria-valuenow', '0');
	        canvas.setAttribute('aria-valuetext', 'No evidence loaded');
	        return;
	      }
	      canvas.tabIndex = 0;
	      const selectedIndex = entries.findIndex((entry) => entry.key === selectedEvidenceKey);
	      const activeIndex = selectedIndex >= 0 ? selectedIndex : entries.length - 1;
	      const activeItem = entries[activeIndex].item;
	      canvas.setAttribute('aria-valuemin', '1');
	      canvas.setAttribute('aria-valuemax', String(entries.length));
	      canvas.setAttribute('aria-valuenow', String(activeIndex + 1));
	      canvas.setAttribute(
	        'aria-valuetext',
	        `${evidenceItemTitle(activeItem)}, ${stageLabel(evidenceItemStage(activeItem))}, ${relativeTime(activeItem.timestamp)}`,
	      );
	    }

		    function requestEvidenceTraceAnimation() {
		      if (evidenceTraceAnimationFrame || evidenceTraceMotionReduced()) return;
		      evidenceTraceAnimationFrame = window.requestAnimationFrame((now) => {
		        evidenceTraceAnimationFrame = 0;
		        drawEvidenceTrace(lastData, now, true);
		      });
		    }

		    function drawEvidenceTrace(data = lastData, now = performance.now(), animationFrame = false) {
		      const canvas = document.getElementById('evidenceTraceCanvas');
		      if (!canvas) return;
		      const context = canvas.getContext('2d');
		      if (!context) return;
		      const entries = traceEntriesForData(data);
		      syncEvidenceTraceAccessibility(canvas, entries);
		      syncEvidenceTraceLegend(entries);
		      const sampledEntries = sampledEvidenceTraceEntries(entries, canvas.getBoundingClientRect().width);
		      canvas.dataset.renderedMarks = String(sampledEntries.length);
		      if (!animationFrame) sizeEvidenceTraceCanvas(canvas);
		      resizeEvidenceTraceCanvas(canvas, context);
	      const rect = canvas.getBoundingClientRect();
	      const width = rect.width;
	      const height = rect.height;
	      const pad = {left: 18, right: 18, top: 20, bottom: 28};
	      evidenceTracePoints = [];
	      context.clearRect(0, 0, width, height);
	      context.fillStyle = canvasToken('--color-panel', '#fbf7ef');
	      context.fillRect(0, 0, width, height);
	      context.strokeStyle = canvasToken('--color-grid-major', 'rgba(25, 23, 18, 0.085)');
	      context.lineWidth = 1;
	      for (let x = 0; x < width; x += 24) {
	        context.beginPath();
	        context.moveTo(x, 0);
	        context.lineTo(x, height);
	        context.stroke();
	      }
	      for (let y = 0; y < height; y += 24) {
	        context.beginPath();
	        context.moveTo(0, y);
	        context.lineTo(width, y);
	        context.stroke();
	      }
		      const mode = evidenceTraceMode();
		      canvas.dataset.mode = mode;
		      canvas.setAttribute('aria-orientation', 'horizontal');
	      if (!entries.length) {
	        drawPreparedTracePlaceholder(context, data, width, height, pad);
	        return;
	      }
		      const animating = drawEvidenceTraceHorizontal(context, sampledEntries, width, height, pad, now);
		      if (animating) requestEvidenceTraceAnimation();
		    }

	    function tracePlaceholderModules(data) {
	      const laneItems = visibleFlowLanes(data)
	        .flatMap((lane) => (lane.items || []).map((item) => ({...item, lane: item.lane || lane.id})))
	        .filter((item) => Number(item.expected_units || item.complete_units || 0) > 0);
	      if (laneItems.length) return laneItems.slice(0, 18);
	      return workingModules(data)
	        .filter((module) => {
	          const stage = canonicalStage(module.lane || module.status || module.lifecycle_state || module.stage);
	          return ['prepared', 'queued', 'generating', 'generation', 'needs_scoring', 'scoring', 'score_ready', 'attention'].includes(stage);
	        })
	        .slice(0, 18);
	    }

	    function drawPreparedTracePlaceholder(context, data, width, height, pad) {
	      const modules = tracePlaceholderModules(data);
	      context.fillStyle = canvasToken('--color-muted', '#6f675d');
	      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
	      context.fillText('prepared contracts', pad.left, 14);
	      if (!modules.length) {
	        context.font = '13px Inter, system-ui, sans-serif';
	        context.fillText('Waiting for saved evidence...', pad.left, Math.max(32, height / 2));
	        return;
	      }
		      const plotWidth = Math.max(1, width - pad.left - pad.right);
		      const baseline = height - 18;
		      const plotHeight = Math.max(24, baseline - pad.top - 10);
		      const step = modules.length > 1 ? plotWidth / (modules.length - 1) : plotWidth;
		      const maxUnits = Math.max(1, ...modules.map((module) => Number(module.expected_units || module.complete_units || 1)));
		      modules.forEach((module, offset) => {
		        const stage = canonicalStage(module.lane || module.stage || 'prepared');
		        const color = evidenceTraceColor(stage === 'prepared' ? 'queued' : stage);
		        const x = pad.left + (modules.length > 1 ? offset * step : plotWidth / 2);
		        const expected = Number(module.expected_units || module.complete_units || 0);
		        const scale = Math.sqrt(Math.max(1, expected) / maxUnits);
		        const barHeight = Math.max(22, Math.round(scale * plotHeight));
		        context.globalAlpha = 0.72;
		        context.strokeStyle = color;
		        context.lineWidth = 2;
		        context.strokeRect(Math.round(x) - 2, baseline - barHeight, 4, barHeight);
		        context.globalAlpha = 1;
		        context.fillStyle = canvasToken('--color-muted', '#6f675d');
		        context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
		        const label = shortHash(module.module || module.title || 'contract');
		        context.textAlign = offset === 0 ? 'left' : (offset === modules.length - 1 ? 'right' : 'center');
		        context.fillText(label, x, baseline + 12);
		      });
	      context.textAlign = 'start';
	      context.globalAlpha = 1;
	    }

		    function drawEvidenceTraceHorizontal(context, entries, width, height, pad, now) {
		      const plotWidth = Math.max(1, width - pad.left - pad.right);
		      const step = entries.length > 1 ? plotWidth / (entries.length - 1) : plotWidth;
		      const laneTop = pad.top + 2;
		      const baseline = height - 10;
		      const laneHeight = Math.max(20, baseline - laneTop);
		      const markWidth = Math.min(4, Math.max(1.5, step * 0.48));
		      const fullBarHeight = Math.max(14, Math.round(laneHeight * EVIDENCE_TRACE_MARK_FRACTION));
		      const ink = canvasToken('--color-ink', '#191712');
		      const lineColor = canvasToken('--color-line', '#b8ad9d');
		      let animating = false;

		      context.fillStyle = canvasToken('--color-muted', '#6f675d');
		      context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
		      context.fillText('older', pad.left, 14);
		      context.strokeStyle = lineColor;
		      context.globalAlpha = 0.48;
		      context.lineWidth = 1;
		      context.beginPath();
		      context.moveTo(pad.left, baseline + 0.5);
		      context.lineTo(width - pad.right, baseline + 0.5);
		      context.stroke();
		      context.globalAlpha = 1;

		      const points = entries.map((entry, offset) => {
		        const item = entry.item;
		        const stage = evidenceItemStage(item);
		        const stageColor = evidenceTraceColor(stage);
		        const moduleColor = evidenceTraceModuleColor(item);
		        const x = pad.left + (entries.length > 1 ? offset * step : plotWidth / 2);
		        return {
		          entry,
		          item,
		          stage,
		          stageColor,
		          moduleColor,
		          x,
		          barHeight: fullBarHeight,
		          progress: 1,
		          fresh: freshEvents.has(evidenceKey(item)),
		        };
		      });
		      const freshPoints = points.filter((point) => point.fresh);
		      const arrivalPoint = freshPoints[freshPoints.length - 1] || null;
		      if (arrivalPoint) {
		        arrivalPoint.progress = evidenceTraceEnterProgress(arrivalPoint.item, now);
		        arrivalPoint.barHeight = fullBarHeight * arrivalPoint.progress;
		        animating = arrivalPoint.progress < 1;
		      }

		      points.forEach((point) => {
		        const {entry, item, stage, stageColor, moduleColor, x, barHeight, progress} = point;
		        const top = baseline - barHeight;
		        const isArrival = point === arrivalPoint;
		        context.save();
		        context.globalAlpha = stage === 'attention' ? 1 : 0.9;
		        if (isArrival && progress < 1) {
		          context.shadowBlur = 12;
		          context.shadowColor = stageColor;
		        }
		        if (stage === 'attention') {
		          context.fillStyle = stageColor;
		          context.fillRect(x - (markWidth + 2) / 2, top, markWidth + 2, barHeight);
		        }
		        context.fillStyle = stageColor;
		        context.fillRect(x - markWidth / 2, top, markWidth, barHeight);
		        context.shadowBlur = 0;
		        context.fillStyle = moduleColor;
		        context.fillRect(x - markWidth / 2, top, markWidth, Math.min(4, barHeight));
		        context.restore();

		        if (entry.key === selectedEvidenceKey && barHeight > 0) {
		          context.strokeStyle = ink;
		          context.lineWidth = 1.5;
		          context.strokeRect(x - markWidth / 2 - 3, top - 3, markWidth + 6, barHeight + 6);
		        }
		        evidenceTracePoints.push({
		          mode: 'horizontal',
		          x,
		          y: baseline - barHeight / 2,
		          width: Math.max(7, step * 0.48),
		          key: entry.key,
		          item,
		        });
		      });

		      if (arrivalPoint) {
		        const pulse = Math.max(0, 1 - arrivalPoint.progress);
		        context.save();
		        context.strokeStyle = arrivalPoint.stageColor;
		        context.globalAlpha = 0.32 + pulse * 0.58;
		        context.lineWidth = 1.5 + pulse * 2.5;
		        context.shadowBlur = pulse * 14;
		        context.shadowColor = arrivalPoint.stageColor;
		        context.beginPath();
		        context.moveTo(arrivalPoint.x, laneTop);
		        context.lineTo(arrivalPoint.x, baseline + 1);
		        context.stroke();
		        context.restore();
		      }

		      const latestPoint = points[points.length - 1];
		      if (latestPoint) {
		        context.globalAlpha = 0.9;
		        context.fillStyle = arrivalPoint ? arrivalPoint.moduleColor : ink;
		        context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
		        context.textAlign = 'right';
		        const arrivalModule = arrivalPoint ? evidenceTraceModule(arrivalPoint.item) : null;
		        const newestLabel = arrivalPoint
		          ? `+${freshPoints.length} new · ${arrivalModule.label}`
		          : 'latest';
		        context.fillText(newestLabel, width - pad.right, 14, plotWidth * 0.58);
		        context.textAlign = 'start';
		        context.globalAlpha = 1;
		      }
		      return animating;
		    }

	    function nearestEvidenceTracePoint(event) {
	      const canvas = document.getElementById('evidenceTraceCanvas');
	      if (!canvas) return null;
	      const rect = canvas.getBoundingClientRect();
	      const x = event.clientX - rect.left;
	      const y = event.clientY - rect.top;
	      let best = null;
	      for (const point of evidenceTracePoints) {
	        const distance = Math.abs(point.x - x);
	        if (distance <= point.width && (!best || distance < best.distance)) {
	          best = {...point, distance, x, y};
	        }
	      }
	      return best;
	    }

	    function showEvidenceTraceTip(point, event) {
	      const tip = document.getElementById('evidenceTraceTip');
	      const canvas = document.getElementById('evidenceTraceCanvas');
	      if (!tip || !canvas) return;
	      if (!point) {
	        tip.hidden = true;
	        return;
	      }
	      const item = point.item;
	      const stage = evidenceItemStage(item);
	      tip.hidden = false;
	      tip.innerHTML = `<strong>${esc(evidenceItemTitle(item))}</strong><div>${esc(stageLabel(stage))} · ${esc(relativeTime(item.timestamp))}</div><div>${esc(evidenceItemMeta(item))}</div><div>${esc(evidenceItemSize(item))} chars saved</div>`;
	      const wrap = canvas.parentElement.getBoundingClientRect();
	      const localX = event.clientX - wrap.left;
	      const scrollTop = canvas.parentElement.scrollTop || 0;
	      const localY = event.clientY - wrap.top + scrollTop;
	      const tipWidth = Math.min(tip.offsetWidth || 240, wrap.width - 32);
	      const tipHeight = tip.offsetHeight || 84;
	      const placeLeft = localX > wrap.width * 0.52;
	      const placeAbove = event.clientY - wrap.top > wrap.height * 0.56;
	      const left = placeLeft
	        ? Math.max(16, localX - tipWidth - 14)
	        : Math.min(wrap.width - tipWidth - 16, localX + 14);
	      const floatingTop = placeAbove ? localY - tipHeight - 18 : localY + 18;
	      const top = Math.min(scrollTop + wrap.height - tipHeight - 12, Math.max(scrollTop + 12, floatingTop));
	      tip.style.left = `${left}px`;
	      tip.style.top = `${top}px`;
	    }

	    function selectEvidenceTraceKey(key, {scroll = false} = {}) {
	      selectedEvidenceKey = key || '';
	      document.querySelectorAll('.feed-card[data-evidence-key]').forEach((card) => {
	        const selected = card.dataset.evidenceKey === selectedEvidenceKey;
	        card.dataset.selected = String(selected);
	        card.setAttribute('aria-current', String(selected));
	      });
	      drawEvidenceTrace(lastData);
	      const canvas = document.getElementById('evidenceTraceCanvas');
	      const wrap = canvas?.parentElement;
	      const tracePoint = evidenceTracePoints.find((point) => point.key === selectedEvidenceKey);
	      if (!scroll || !selectedEvidenceKey) return;
	      const feed = document.getElementById('evidenceFeed');
	      if (!feed) return;
	      let target = null;
	      for (const card of feed.querySelectorAll('.feed-card[data-evidence-key]')) {
	        if (card.dataset.evidenceKey === selectedEvidenceKey) {
	          target = card;
	          break;
	        }
	      }
	      if (!target) return;
	      evidenceAutoFollow = false;
	      evidenceTraceAutoFollow = false;
	      const feedRect = feed.getBoundingClientRect();
	      const targetRect = target.getBoundingClientRect();
	      const targetTop = feed.scrollTop + targetRect.top - feedRect.top - Math.max(8, (feed.clientHeight - targetRect.height) / 2);
	      feed.scrollTo({top: Math.max(0, targetTop), behavior: 'auto'});
	      syncPersistentControls();
	    }

	    function setEvidenceTraceWindow(value) {
	      const next = value === 'all' ? 'all' : String(Math.max(1, Math.round(Number(value) || 50)));
	      evidenceTraceWindow = next;
	      selectedEvidenceKey = '';
	      evidenceAutoFollow = true;
	      evidenceTraceAutoFollow = true;
	      pendingEvidenceLiveSnap = true;
	      syncEvidenceWindowButtons();
	      if (lastData) render(lastData);
	    }

	    function setEvidenceContentFilter(value) {
	      const allowed = new Set(['text', 'all', 'writes']);
	      evidenceContentFilter = allowed.has(value) ? value : 'all';
	      selectedEvidenceKey = '';
	      syncEvidenceContentButtons();
	      if (lastData) render(lastData);
	    }

	    function bindEvidenceTrace() {
	      const canvas = document.getElementById('evidenceTraceCanvas');
	      if (!canvas || canvas.dataset.boundEvidenceTrace === 'true') return;
	      canvas.dataset.boundEvidenceTrace = 'true';
	      canvas.addEventListener('mousemove', (event) => {
	        showEvidenceTraceTip(nearestEvidenceTracePoint(event), event);
	      });
	      canvas.addEventListener('mouseleave', () => {
	        const tip = document.getElementById('evidenceTraceTip');
	        if (tip) tip.hidden = true;
	      });
	      canvas.addEventListener('click', (event) => {
	        const point = nearestEvidenceTracePoint(event);
	        if (point) selectEvidenceTraceKey(point.key, {scroll: true});
	      });
	      canvas.addEventListener('keydown', (event) => {
	        const entries = traceEntriesForData(lastData);
	        if (!entries.length) return;
	        const selectedIndex = entries.findIndex((entry) => entry.key === selectedEvidenceKey);
	        const currentIndex = selectedIndex >= 0 ? selectedIndex : entries.length - 1;
	        let nextIndex = currentIndex;
	        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex -= 1;
	        else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex += 1;
	        else if (event.key === 'Home') nextIndex = 0;
	        else if (event.key === 'End') nextIndex = entries.length - 1;
	        else return;
	        event.preventDefault();
	        nextIndex = Math.max(0, Math.min(entries.length - 1, nextIndex));
	        selectEvidenceTraceKey(entries[nextIndex].key, {scroll: true});
	      });
	    }

	    function stageFilterButton(stage, label) {
	      const active = activeStageFilter === stage;
	      const classes = stage === 'all'
	        ? 'stage-filter stage-filter-all stage-filter-clear'
	        : `stage-filter ${stageClass(stage)}`;
	      return `
	        <button class="${classes}" type="button" data-stage-filter="${esc(stage)}" aria-pressed="${active}">
	          <span class="stage-dot" aria-hidden="true"></span>${esc(label)}
	        </button>
	      `;
	    }

	    function renderEvidenceItem(item, key = evidenceKey(item)) {
	      const stage = evidenceItemStage(item);
	      const title = evidenceItemTitle(item);
	      const meta = evidenceItemMeta(item);
	      const modelChip = item.model ? renderModelChip(item.model, 'compact') : '';
	      const fresh = freshEvents.has(evidenceKey(item)) ? ' is-new' : '';
	      const selected = selectedEvidenceKey === key;
	      if (item.kind === 'turn_pair') {
	        const checks = item.checks || {};
	        return `
	          <article class="feed-card ${stageClass(stage)}${fresh}" data-evidence-key="${esc(key)}" data-selected="${selected}" aria-current="${selected}" tabindex="0">
	            <div class="feed-top">
	              <div class="feed-title">${esc(title)}</div>
	              <div class="model-stack">${modelChip}<span class="stage-pill"><span class="stage-dot"></span>${esc(stageLabel(stage))}</span></div>
	            </div>
	            <div class="feed-meta">${esc(meta)} · ${esc(relativeTime(item.timestamp))}</div>
	            <div class="qa-pair">
	              <div class="qa-block">
	                <div class="qa-label">Question / user pressure</div>
	                <div class="qa-text markdown-text">${renderMarkdown(unwrapMessageContent(item.user_message || 'No saved user text.'))}</div>
	              </div>
	              <div class="qa-block answer">
	                <div class="qa-label">Answer / model response</div>
	                <div class="qa-text markdown-text">${renderMarkdown(unwrapMessageContent(item.model_response || 'No saved assistant text.'))}</div>
	              </div>
	            </div>
	            <div class="check-strip">
	              ${renderCheck('JSON', checks.json_ok !== false)}
	              ${renderCheck('user', Boolean(checks.user_ok))}
	              ${renderCheck('answer', Boolean(checks.assistant_ok))}
	              ${renderCheck('turn', Boolean(checks.turn_ok))}
	              ${renderCheck('provider text', !checks.provider_error)}
	            </div>
	            ${item.problem ? `<div class="feed-problem">${esc(item.problem)}</div>` : ''}
	            <div class="feed-meta">${esc(item.transcript_path || '')}</div>
	          </article>
	        `;
	      }
	      const classificationCopy = item.evidence_class
	        ? `${item.evidence_class} / ${item.category || 'unclassified'}${item.action ? ` · ${item.action}` : ''}`
	        : '';
	      const eventCopy = [classificationCopy, item.problem || judgeResultCopy(item) || eventTitle(item)].filter(Boolean).join('\n\n');
	      return `
	        <article class="feed-card ${stageClass(stage)}${fresh}" data-evidence-key="${esc(key)}" data-selected="${selected}" aria-current="${selected}" tabindex="0">
	          <div class="feed-top">
	            <div class="feed-title">${esc(title)}</div>
	            <div class="model-stack">${modelChip}<span class="stage-pill"><span class="stage-dot"></span>${esc(stageLabel(stage))}</span></div>
	          </div>
	          <div class="feed-meta">${esc(meta)} · ${esc(relativeTime(item.timestamp))}</div>
	          <div class="event-feed-copy markdown-text">${renderMarkdown(eventCopy)}</div>
	          ${item.score_path ? `<div class="feed-meta">${esc(item.score_path)}</div>` : ''}
	          ${item.transcript_path ? `<div class="feed-meta">${esc(item.transcript_path)}</div>` : ''}
	        </article>
	      `;
	    }

	    function renderEvidenceFeed(data) {
	      const items = data.evidence_feed || [];
	      const filteredEntries = filteredEvidenceEntries(items);
	      const visibleEntries = windowedEvidenceEntries(filteredEntries);
	      const feedEntries = visibleEntries.slice().reverse();
	      const totalEntries = Number(data.evidence_total_count ?? filteredEntries.length);
	      const filterLabel = activeStageFilter === 'all' ? 'All stages' : stageLabel(activeStageFilter);
	      const inspectionMode = activeStageFilter !== 'all';
	      const countLabel = visibleEntries.length < totalEntries
	        ? `${visibleEntries.length} / ${totalEntries} entries`
	        : `${visibleEntries.length} entries`;
	      const emptyCopy = data.evidence_loading
	        ? 'Loading scoped evidence...'
	        : data.evidence_detail_error
	          ? 'Evidence detail is temporarily unavailable.'
	          : `No ${filterLabel.toLowerCase()} evidence found yet. Waiting for saved turns, score files, or RUN_EVENTS.jsonl writes.`;
	      return `
	        <section class="panel feed-panel feed-shell${inspectionMode ? ' is-inspection' : ''}">
	          <div class="panel-head">
	            <div>
	              <h2>Live evidence feed</h2>
	              <div class="panel-kicker">Saved turns and judge writes for the selected scope.</div>
	            </div>
	            <div class="run-meta">
	              <span class="chip">${esc(runScopeLabel(data))}</span>
	              <span class="chip">${esc(evidenceContentLabel())}</span>
	              <span class="chip">${esc(filterLabel)}</span>
	              <span class="chip">${esc(evidenceWindowLabel())}</span>
	              <span class="chip">${esc(countLabel)}</span>
	              <button class="copy-button" type="button" data-jump-live id="evidenceLiveButton">Jump to live</button>
	            </div>
	          </div>
	          <div class="feed-toolbar">
	            <div class="feed-filter-group" role="group" aria-label="Evidence stage filter">
	              ${stageFilterButton('all', 'All')}
	              ${stageFilterButton('generating', 'Generating')}
	              ${stageFilterButton('judging', 'Judging')}
	              ${stageFilterButton('score_ready', 'Scored')}
	              ${stageFilterButton('attention', 'Attention')}
	            </div>
	            <div class="evidence-window-controls" role="group" aria-label="Evidence content view">
	              <span class="trace-window-label">View</span>
	              ${evidenceContentButton('text', 'Text')}
	              ${evidenceContentButton('all', 'All')}
	              ${evidenceContentButton('writes', 'Writes')}
	            </div>
	            <div class="evidence-window-controls" role="group" aria-label="Evidence trace window">
	              <span class="trace-window-label">Window</span>
	              ${evidenceTraceWindowButton('all', 'All')}
	              ${evidenceTraceWindowButton('100', '100')}
	              ${evidenceTraceWindowButton('50', '50')}
	              ${evidenceTraceWindowButton('25', '25')}
	            </div>
	          </div>
		          <div class="evidence-trace-legend" id="evidenceTraceLegend" aria-label="Evidence trace legend"></div>
		          <div class="evidence-trace-wrap">
		            <canvas class="evidence-trace-canvas" id="evidenceTraceCanvas" role="slider" tabindex="0" aria-label="Evidence event trace; line color indicates stage and top marker indicates benchmark module" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" aria-valuetext="No evidence loaded"></canvas>
	            <div class="evidence-trace-tip" id="evidenceTraceTip" hidden></div>
	          </div>
	          <div class="feed-scroll${inspectionMode ? ' is-inspection' : ''}" id="evidenceFeed" aria-live="polite" aria-label="Evidence reader">
	            ${feedEntries.map((entry) => renderEvidenceItem(entry.item, entry.key)).join('') || `<div class="empty">${esc(emptyCopy)}</div>`}
	          </div>
	        </section>
	      `;
	    }

		    function capacitySourceLabel(data) {
		      const source = String(data?.paid_call_leases?.capacity?.effective_limit_source || '');
		      if (source.startsWith('environment:')) return '.env ceiling';
		      if (source === 'policy:operator') return 'operator policy';
		      if (source.startsWith('policy:')) return 'saved policy';
		      if (source === 'default') return 'built-in default';
		      return 'unavailable';
		    }

	    function renderRadar(data) {
	      const modules = workingRawModules(data)
	        .filter((module) => module.attention || module.severity === 'attention')
	        .filter((module) => showAcknowledged || !isHiddenModule(module))
	        .sort(sortByActivity)
	        .slice(0, 5);
	      const issueCards = modules.map((module) => {
	        const attention = module.attention || {};
	        const reason = attention.reason || (module.spend_guard || {}).reason || 'Inspect this ledger before continuing.';
	        const next = attention.action || (module.score_state || {}).action || 'Inspect RUN_STATUS.json and RUN_EVENTS.jsonl.';
	        const ackKey = moduleAckKey(module);
	        const acknowledgedModule = isAcknowledgedModule(module);
	        const rejectedModule = isRejectedModule(module);
	        return `
	          <article class="radar-card attention ${(acknowledgedModule || rejectedModule) ? 'acknowledged' : ''}">
	            <div class="radar-title">${esc(module.run_id)} / ${esc(module.module_path || module.module)}</div>
	            <div class="stage-line">${esc(attention.title || module.status || 'Needs review')} · ${esc(module.elapsed || 'elapsed unknown')}</div>
	            ${failureClassificationMarkup(attention)}
	            <div class="failure-detail"><div class="stat-label">Recorded failure</div><div>${esc(reason)}</div></div>
	            <div class="stage-line">${esc(next)}</div>
	            <div class="action-row">
	              ${module.status_path ? `<button class="copy-button danger" type="button" data-disposition="${esc(module.status_path)}" data-disposition-action="reject">Reject from analysis</button>` : ''}
	              ${ackKey ? `<button class="copy-button" type="button" data-ack="${esc(ackKey)}">${acknowledgedModule ? 'Unhide' : 'Hide locally'}</button>` : ''}
	            </div>
	          </article>
	        `;
	      }).join('');
	      const unknown = scopedUnknownCostSummary(data);
	      const provenanceCard = unknown.calls ? `
	        <article class="radar-card warning">
	          <div class="radar-title">Spend provenance</div>
	          <div class="stage-line">Accounting warning; benchmark evidence remains intact.</div>
	          <div class="provenance-detail">
	            <div class="stat-label">Unpriced recorded calls</div>
	            <div><strong>${esc(unknown.calls)}</strong> across ${esc(unknown.modules)} module${unknown.modules === 1 ? '' : 's'}</div>
	          </div>
	          <div class="stage-line">Add pricing metadata before publishing spend totals. This is not a recorded model or run failure.</div>
	        </article>
	      ` : '';
	      return `
	        <aside class="panel radar-panel">
	          <div class="panel-head">
	            <div>
	              <h2>Radar</h2>
	              <div class="panel-kicker">Run integrity and spend provenance in ${esc(runScopeLabel(data))}.</div>
	            </div>
	          </div>
	          <div class="radar-body">
	            ${issueCards || '<article class="radar-card"><div class="radar-title">No active blockers</div><div class="stage-line">No failed, stale, rate-limited, or malformed modules in the working view.</div></article>'}
	            ${provenanceCard}
	          </div>
	        </aside>
	      `;
	    }

	    function captureEvidenceFeedState() {
	      const feed = document.getElementById('evidenceFeed');
	      if (!feed) return {atLive: true, scrollTop: 0, scrollHeight: 0, pageY: window.scrollY};
	      return {
	        atLive: feed.scrollTop <= 36,
	        scrollTop: feed.scrollTop,
	        scrollHeight: feed.scrollHeight,
	        pageY: window.scrollY,
	      };
	    }

	    function syncEvidenceFeedScroll(previous, {evidenceChanged = true} = {}) {
	      const feed = document.getElementById('evidenceFeed');
	      const button = document.getElementById('evidenceLiveButton');
	      if (!feed) return;
	      const update = () => {
	        if (activeStageFilter !== 'all') evidenceAutoFollow = false;
	        else evidenceAutoFollow = feed.scrollTop <= 36;
	        evidenceTraceAutoFollow = evidenceAutoFollow;
	        if (button) button.hidden = evidenceAutoFollow;
	        syncPersistentControls();
	      };
	      if (pendingEvidenceLiveSnap) {
	        pendingEvidenceLiveSnap = false;
	        feed.scrollTop = 0;
	        scrollEvidenceTraceToLive();
	      } else if (pendingFeedPanelScroll) {
	        pendingFeedPanelScroll = false;
	        feed.scrollTop = 0;
	        scrollToFeedPanel();
	      } else if (evidenceChanged && evidenceAutoFollow && previous.atLive) {
	        scrollFeedToLive();
	        scrollEvidenceTraceToLive();
	      } else {
	        const heightDelta = evidenceChanged ? Math.max(0, feed.scrollHeight - (previous.scrollHeight || 0)) : 0;
	        feed.scrollTop = Math.min(previous.scrollTop + heightDelta, Math.max(0, feed.scrollHeight - feed.clientHeight));
	        window.scrollTo({top: previous.pageY, behavior: 'auto'});
	      }
	      update();
	      bindEvidenceFeedBoundaryScroll(feed, update);
	    }

	    function scrollFeedToLive() {
	      const feed = document.getElementById('evidenceFeed');
	      if (!feed) return;
	      feed.scrollTop = 0;
	    }

	    function bindEvidenceFeedBoundaryScroll(feed, update) {
	      if (!feed || feed.dataset.boundaryScroll === 'true') return;
	      feed.dataset.boundaryScroll = 'true';
	      feed.addEventListener('scroll', update, {passive: true});
	      const pauseFollow = () => {
	        evidenceAutoFollow = false;
	        evidenceTraceAutoFollow = false;
	        const button = document.getElementById('evidenceLiveButton');
	        if (button) button.hidden = false;
	        syncPersistentControls();
	      };
	      feed.addEventListener('pointerdown', pauseFollow, {passive: true});
	      feed.addEventListener('touchstart', pauseFollow, {passive: true});
	      feed.addEventListener('wheel', (event) => {
	        pauseFollow();
	        if (feed.scrollHeight <= feed.clientHeight) return;
	        const atTop = feed.scrollTop <= 0;
	        const atBottom = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 1;
	        const leavingTop = event.deltaY < 0 && atTop;
	        const leavingBottom = event.deltaY > 0 && atBottom;
	        if (!leavingTop && !leavingBottom) return;
	        event.preventDefault();
	        window.scrollBy({top: event.deltaY, behavior: 'auto'});
	      }, {passive: false});
	    }

	    function scrollToFeedPanel() {
	      const panel = document.querySelector('.feed-panel');
	      if (panel) panel.scrollIntoView({behavior: 'auto', block: 'start'});
	    }

	    function syncPersistentControls() {
	      document.querySelectorAll('[data-quick-stage]').forEach((button) => {
	        button.setAttribute('aria-pressed', String(activeStageFilter === (button.dataset.quickStage || 'all')));
	      });
	      const followButton = document.getElementById('followLiveToggle');
	      if (followButton) {
	        followButton.setAttribute('aria-pressed', String(evidenceAutoFollow));
	        followButton.textContent = evidenceAutoFollow ? 'Pause feed' : 'Follow live';
	      }
	      syncEvidenceWindowButtons();
	      syncEvidenceContentButtons();
	    }

	    function applyStageFilter(next, scrollFeed = false) {
	      const normalized = canonicalStage(next || 'all');
	      activeStageFilter = activeStageFilter === normalized && normalized !== 'all' ? 'all' : normalized;
	      selectedEvidenceKey = '';
	      if (scrollFeed || normalized !== 'all') {
	        evidenceAutoFollow = false;
	        evidenceTraceAutoFollow = false;
	      }
	      if (scrollFeed) pendingFeedPanelScroll = true;
	      if (lastData) render(lastData);
	    }

	    function setEvidenceFollow(next) {
	      evidenceAutoFollow = Boolean(next);
	      evidenceTraceAutoFollow = Boolean(next);
	      if (evidenceAutoFollow) {
	        pendingEvidenceLiveSnap = true;
	        scrollFeedToLive();
	        scrollEvidenceTraceToLive();
	      }
	      const button = document.getElementById('evidenceLiveButton');
	      if (button) button.hidden = evidenceAutoFollow;
	      syncPersistentControls();
	    }

	    function eventTickClass(event) {
      const name = String(event.event || '');
      if (name.includes('failed') || name.includes('incomplete')) return 'failed';
      if (name.includes('score')) return 'score';
      if (name.includes('turn') || name.includes('conversation')) return 'turn';
      return 'gap';
    }

    function primaryModule(data) {
      const modules = allModules(data);
      if (!modules.length) return null;
      const sorted = [...modules].sort(sortByActivity);
      return (
        sorted.find((module) => module.severity === 'running' && (module.latest_transcript || {}).model_response) ||
        sorted.find((module) => module.severity === 'running') ||
        sorted.find((module) => module.severity === 'attention') ||
        sorted.find((module) => (module.latest_transcript || {}).model_response) ||
        sorted[0]
      );
    }

    function renderPrimaryModule(data) {
      const module = primaryModule(data);
      if (!module) return '';
      const progress = module.progress || {};
      const transcript = module.latest_transcript || {};
      const percent = progress.percent;
      const width = percent == null ? 0 : Math.max(0, Math.min(100, Number(percent)));
      const percentLabel = percent == null ? 'event' : `${percent}%`;
      const turns = turnCountLabel(progress);
      const conversations = conversationCountLabel(progress);
      const latest = module.latest_event ? `${eventTitle(module.latest_event)} | ${relativeTime(module.latest_event.timestamp)}` : 'No events yet';
      const cost = module.cost ? money(module.cost.total_cost_usd) : 'not tracked';
      const transcriptMeta = [
        module.module,
        transcript.model,
        transcript.test_type,
        transcript.item_idx != null ? `item ${transcript.item_idx}` : '',
        transcript.side,
        transcript.turn ? `turn ${transcript.turn}` : '',
      ].filter(Boolean).join(' | ');
      const recent = (module.recent_events || []).slice(-24);
      const bars = recent.length
        ? recent.map((event) => `<span class="event-tick ${eventTickClass(event)}" title="${esc(eventTitle(event))}"></span>`).join('')
        : '<span class="event-tick gap" title="No ledger writes yet"></span>';
      return `
        <section class="panel primary-watch ${esc(module.severity)}">
          <div class="panel-head">
            <div>
              <h2>Primary Module Watch</h2>
              <div class="panel-kicker">The run that needs eyes first: event progress, newest artifact write, and the latest saved turn text.</div>
            </div>
            <div class="run-meta">
              ${statusBadge(module)}
              <span class="chip">${esc(module.stage || 'stage unknown')}</span>
              <span class="chip">poll: ${esc((pollMs / 1000).toFixed(1))}s</span>
            </div>
          </div>
          <div class="primary-body">
            <div class="primary-main">
              <div>
                <div class="primary-eyebrow">${esc(module.module || 'module')} · ${esc(module.stage || 'stage')}</div>
                <div class="primary-title">${esc(module.run_id)} / ${esc(module.module_path || module.module)}</div>
              </div>
              <div>
                <div class="primary-percent">${esc(percentLabel)}</div>
                <div class="primary-progress-copy">${percent == null ? 'Progress is ledger-derived.' : `full-ledger progress · ${turns}`}</div>
              </div>
              <div class="bar"><div class="fill" style="width:${width}%"></div></div>
              <div class="primary-stats">
                <div class="primary-stat"><div class="stat-label">Turns</div><div class="stat-value">${esc(turns)}</div></div>
                <div class="primary-stat"><div class="stat-label">Conversations</div><div class="stat-value">${esc(conversations)}</div></div>
                <div class="primary-stat"><div class="stat-label">Scores</div><div class="stat-value">${esc(progress.scores_saved || 0)}</div></div>
                <div class="primary-stat"><div class="stat-label">Cost</div><div class="stat-value">${esc(cost)}</div></div>
              </div>
	              <div class="primary-turns">
	                <div class="turn-card">
	                  <div class="stat-label">Latest user pressure</div>
	                  <div class="turn-text markdown-text">${renderMarkdown(unwrapMessageContent(transcript.user_message || 'No saved user turn yet.'))}</div>
	                </div>
	                <div class="turn-card model">
	                  <div class="stat-label">Latest model response</div>
	                  ${transcript.model ? renderModelChip(transcript.model, 'compact') : ''}
	                  <div class="turn-text markdown-text">${renderMarkdown(unwrapMessageContent(transcript.model_response || 'No saved model response yet.'))}</div>
	                  <div class="event-detail">${esc(transcriptMeta)}</div>
	                </div>
	              </div>
            </div>
            <aside class="primary-side">
              <div class="event-tape">
                <div class="module-top">
                  <div>
                    <div class="stat-label">Recent ledger writes</div>
                    <div class="stage-line">${esc(latest)}</div>
                  </div>
                  <span class="chip">${esc(recent.length)} shown</span>
                </div>
                <div class="event-bars">${bars}</div>
                <div class="event-legend">
                  <span class="legend-dot turn">turn/conversation</span>
                  <span class="legend-dot score">score</span>
                  <span class="legend-dot slow">slow/other</span>
                  <span class="legend-dot failed">failed</span>
                </div>
              </div>
              <div class="score-state ${esc((module.score_state || {}).kind || '')}">
                <div class="stat-label">Score state</div>
                <div>${esc((module.score_state || {}).label || 'unknown')}</div>
                <div class="stage-line">${esc((module.score_state || {}).action || '')}</div>
              </div>
            </aside>
          </div>
        </section>
      `;
    }

    function renderActivityCard(module) {
      const progress = module.progress || {};
      const percent = progress.percent;
      const width = percent == null ? 0 : Math.max(0, Math.min(100, Number(percent)));
      const latest = module.latest_event ? `${eventTitle(module.latest_event)} | ${relativeTime(module.latest_event.timestamp)}` : 'No events yet';
      const transcript = module.latest_transcript || {};
      const cost = module.cost ? money(module.cost.total_cost_usd) : 'not tracked';
      const turns = turnCountLabel(progress);
      const conversations = conversationCountLabel(progress);
      const transcriptMeta = [
        transcript.model,
        transcript.test_type,
        transcript.item_idx != null ? `item ${transcript.item_idx}` : '',
        transcript.side,
        transcript.turn ? `turn ${transcript.turn}` : '',
      ].filter(Boolean).join(' | ');
      return `
        <article class="activity-card ${esc(module.severity)}">
          <div class="module-top">
            <div>
              <div class="activity-title">${esc(module.run_id)} / ${esc(module.module_path || module.module)}</div>
              <div class="stage-line">${esc(module.stage || 'unknown')} | elapsed ${esc(module.elapsed || 'unknown')} | cost ${esc(cost)}</div>
            </div>
            ${statusBadge(module)}
          </div>
          <div class="bar"><div class="fill" style="width:${width}%"></div></div>
          <div class="stage-line">${percent == null ? 'Progress is ledger-based' : `Progress ${percent}%`} | turns ${esc(turns)} | conversations ${esc(conversations)}</div>
          <div class="activity-event"><strong>Latest:</strong> ${esc(latest)}</div>
	          <div class="activity-preview">
	            <div class="stat-label">Latest saved turn</div>
	            ${transcript.model_response
	              ? `<div class="model-stack">${transcript.model ? renderModelChip(transcript.model, 'compact') : ''}</div><div class="activity-answer markdown-text">${renderMarkdown(unwrapMessageContent(transcript.model_response))}</div><div class="event-detail">${esc(transcriptMeta)}<br>${esc(transcript.path || '')}</div>`
	              : '<div class="activity-event">No saved transcript turns yet. Waiting for the next artifact write.</div>'}
	          </div>
        </article>
      `;
    }

    function renderLiveActivity(data) {
      const modules = workingModules(data).sort(sortByActivity);
      const running = modules.filter((module) => module.severity === 'running');
      const shown = (running.length ? running : modules).slice(0, 6);
      if (!shown.length) return '<div class="empty">No run activity found yet.</div>';
      return shown.map(renderActivityCard).join('');
    }

    function flowProgressLabel(item) {
      const expected = item.expected_units;
      const complete = item.complete_units;
      if (expected != null) return `${complete || 0} / ${expected} units`;
      const planned = item.planned_turns;
      if (planned) return `${item.turn_saved || 0} / ${planned} turns`;
      if (item.turn_saved) return `${item.turn_saved} turns`;
      return `${Math.max(0, Math.min(100, Number(item.progress_percent || 0)))}%`;
    }

    function renderFlowItem(item) {
      const acknowledgedItem = isAcknowledgedFlowItem(item);
      const rejectedItem = isRejectedFlowItem(item);
      const ackKey = flowAckKey(item);
      const statusPath = item.status_path || '';
      const width = Math.max(0, Math.min(100, Number(item.progress_percent || 0)));
      const cost = item.cost_total_usd != null ? money(item.cost_total_usd) : '';
      const latest = item.latest_event ? relativeTime(item.latest_event.timestamp) : '';
      const conditionHash = item.benchmark_condition_hash ? shortHash(item.benchmark_condition_hash) : '';
      const runsetHash = item.comparison_spec_hash ? shortHash(item.comparison_spec_hash) : '';
      const title = item.title || `${item.run_id || 'run'} / ${item.module_path || item.module || 'module'}`;
	      const modelChips = renderModelStack(item.model_names || [], 5);
	      const latestText = item.latest_model_response || item.latest_user_message || '';
      const eta = item.scheduler_eta_seconds != null ? duration(item.scheduler_eta_seconds) : '';
      const schedulerMeta = [
        item.scheduler_state ? `scheduler: ${item.scheduler_state}` : '',
        item.max_active_calls ? `budget ${item.max_active_calls}` : '',
        item.scheduler_active_units != null ? `active ${item.scheduler_active_units}` : '',
        eta ? `ETA ${eta}` : '',
      ].filter(Boolean).join(' | ');
      return `
        <article class="flow-item ${esc(item.lane || '')} ${(acknowledgedItem || rejectedItem) ? 'acknowledged' : ''}">
          <div class="flow-item-title">${esc(title)}</div>
          <div class="flow-run">${esc(item.run_id || '')}</div>
          <div class="flow-mini">
            <span>${esc(item.model_summary || 'model set')}</span>
            <span>judge: ${esc(item.judge_summary || 'unknown')}</span>
            <span>${esc(item.stage || 'stage')}</span>
            <span>${esc(item.status || 'status')}</span>
            ${cost ? `<span>${esc(cost)}</span>` : ''}
            ${latest ? `<span>${esc(latest)}</span>` : ''}
            ${eta ? `<span>${esc(`ETA ${eta}`)}</span>` : ''}
            ${conditionHash ? `<code title="${esc(item.benchmark_condition_hash || '')}">condition ${esc(conditionHash)}</code>` : ''}
            ${runsetHash ? `<code title="${esc(item.comparison_spec_hash || '')}">runset ${esc(runsetHash)}</code>` : ''}
          </div>
          ${schedulerMeta ? `<div class="flow-run">${esc(schedulerMeta)}</div>` : ''}
	          ${modelChips ? `<div class="flow-models">${modelChips}</div>` : ''}
	          <div class="bar"><div class="fill" style="width:${width}%"></div></div>
	          <div class="flow-mini"><span>${esc(flowProgressLabel(item))}</span><span>${esc(item.validity || '')}</span></div>
	          ${latestText ? `<div class="flow-snippet markdown-text">${renderMarkdown(unwrapMessageContent(latestText))}</div>` : ''}
          <div class="flow-action">${esc(item.next_action || '')}</div>
          <div class="action-row">
            ${item.execute_command ? `<button class="copy-button" type="button" data-copy="${esc(item.execute_command)}">Copy scheduler command</button>` : ''}
            ${item.contract_path ? `<button class="copy-button" type="button" data-copy="${esc(item.contract_path)}">Copy contract</button>` : ''}
            ${item.benchmark_condition_hash ? `<button class="copy-button" type="button" data-copy="${esc(item.benchmark_condition_hash)}">Copy condition</button>` : ''}
            ${item.lane === 'attention' && statusPath ? `<button class="copy-button danger" type="button" data-disposition="${esc(statusPath)}" data-disposition-action="reject">Reject from analysis</button>` : ''}
            ${rejectedItem && statusPath ? `<button class="copy-button" type="button" data-disposition="${esc(statusPath)}" data-disposition-action="restore">Restore diagnostic</button>` : ''}
            ${item.lane === 'attention' && ackKey ? `<button class="copy-button" type="button" data-ack="${esc(ackKey)}">${acknowledgedItem ? 'Unhide' : 'Hide locally'}</button>` : ''}
          </div>
        </article>
      `;
    }

    function renderRunFlow(data) {
      const lanes = visibleFlowLanes(data);
      if (!lanes.length) return '';
      const workGroups = (lane) => lane.work_group_count ?? lane.group_count ?? 0;
      const laneHeadline = (lane) => `${workGroups(lane)} work group${workGroups(lane) === 1 ? '' : 's'}`;
      const laneMeta = (lane) => {
        const cards = `${lane.count || 0} module card${(lane.count || 0) === 1 ? '' : 's'}`;
        const expected = Number(lane.expected_units || 0);
        const complete = Number(lane.complete_units || 0);
        if (lane.id === 'prepared' && expected) return `${cards} · ${expected} planned work units`;
        const units = expected ? `${complete} / ${expected} work units` : `${lane.unit_count || 0} work units`;
        return `${cards} · ${units}`;
      };
      const cards = lanes.map((lane) => `
        <section class="flow-lane" aria-label="${esc(lane.title || lane.id)}">
          <div class="flow-lane-head">
            <div class="flow-lane-title"><span>${esc(lane.title || lane.id)}</span><span class="chip">${esc(laneHeadline(lane))}</span></div>
            <div class="flow-lane-copy">${esc(lane.description || '')}</div>
            <div class="flow-lane-copy">${esc(laneMeta(lane))}</div>
          </div>
          <div class="flow-items">
            ${(lane.items || []).slice(0, 4).map(renderFlowItem).join('') || '<div class="flow-empty">No modules in this lane.</div>'}
          </div>
        </section>
      `).join('');
      return `<div class="flow-board">${cards}</div>`;
    }

    function scopedContractSummary(data) {
      const resolvedScope = resolvedRunScope(data);
      if (data.contract_detail_scope === resolvedScope && data.contract_detail_summary) {
        return data.contract_detail_summary;
      }
      if (resolvedScope === 'all') {
        const summary = data.summary || {};
        return {
          count: Number(summary.contract_count || 0),
          complete_units: Number(summary.contract_complete_units || 0),
          expected_units: Number(summary.contract_expected_units || 0),
          attention_count: Number(summary.contract_attention_count || 0),
          active_control_count: Number(summary.active_control_count || 0),
        };
      }
      const members = familyMemberIds(evidenceRunScope, data);
      if (members.length) {
        const empty = {count: 0, complete_units: 0, expected_units: 0, attention_count: 0, active_control_count: 0};
        return (data.groups || [])
          .filter((item) => members.includes(String(item.run_id || '')))
          .reduce((acc, item) => {
            const cs = item.contract_summary || {};
            for (const key of Object.keys(empty)) acc[key] += Number(cs[key] || 0);
            return acc;
          }, {...empty});
      }
      const group = (data.groups || []).find((item) => String(item.run_id || '') === String(resolvedScope));
      return group?.contract_summary || {count: 0, complete_units: 0, expected_units: 0, attention_count: 0, active_control_count: 0};
    }

    function estimateRangeText(range) {
      if (!range || typeof range !== 'object') return 'unavailable';
      const low = money(range.low);
      const expected = money(range.expected);
      const high = money(range.high);
      return `${expected} expected · ${low}–${high} range`;
    }

    function renderContractCostEstimate(contract) {
      const callPlan = contract.call_plan || {};
      const estimate = contract.cost_estimate || {};
      const calls = callPlan.total_calls || {};
      const expectedCalls = Number(calls.expected || 0);
      const callRange = calls.low != null && calls.high != null
        ? `${Number(calls.low || 0)}–${Number(calls.high || 0)} calls`
        : expectedCalls ? `${expectedCalls} calls` : 'call count unavailable';
      if (!estimate.state) {
        return `
          <div class="contract-estimate">
            <div>
              <div class="stat-label">Planned API work</div>
              <div class="stat-value">${esc(callRange)}</div>
            </div>
            <div class="stage-line">Pricing snapshot not attached. Planning estimate only; no provider calls were made.</div>
          </div>
        `;
      }
      const byStage = estimate.cost_by_stage || {};
      const byRole = estimate.cost_by_role || {};
      const byProvider = estimate.cost_by_provider || {};
      const providerRows = Object.entries(byProvider).map(([provider, range]) => (
        `<span class="chip">${esc(provider)} ${esc(estimateRangeText(range))}</span>`
      )).join('');
      const unknown = (estimate.unknown_pricing || []).length
        ? `<div class="reason">Unknown pricing: ${esc((estimate.unknown_pricing || []).join(', '))}</div>`
        : '';
      return `
        <div class="contract-estimate">
          <div class="contract-estimate-total">
            <div class="stat-label">Estimated run cost</div>
            <div class="stat-value">${esc(estimateRangeText(estimate.total_cost_usd))}</div>
            <div class="stage-line">${esc(callRange)} · ${esc(estimate.state)}</div>
          </div>
          <div>
            <div class="stat-label">Generation estimate</div>
            <div class="stat-value compact">${esc(estimateRangeText(byStage.generation))}</div>
          </div>
          <div>
            <div class="stat-label">Scoring estimate</div>
            <div class="stat-value compact">${esc(estimateRangeText(byStage.scoring))}</div>
          </div>
          <div>
            <div class="stat-label">Model-under-test estimate</div>
            <div class="stat-value compact">${esc(estimateRangeText(byRole.model_under_test))}</div>
          </div>
          <div>
            <div class="stat-label">Run-time support estimate</div>
            <div class="stat-value compact">${esc(estimateRangeText(byRole.support))}</div>
          </div>
          <div>
            <div class="stat-label">Judge-call estimate</div>
            <div class="stat-value compact">${esc(estimateRangeText(byRole.judge))}</div>
          </div>
          ${providerRows ? `<div class="contract-estimate-providers">${providerRows}</div>` : ''}
          ${unknown}
          <div class="stage-line">Planning estimate only; provider-reported usage and invoices are authoritative.</div>
        </div>
      `;
    }

    function renderContracts(data) {
      const resolvedScope = resolvedRunScope(data);
      const detailLoaded = Boolean(data.contract_detail_key) && data.contract_detail_scope === resolvedScope;
      const contracts = detailLoaded ? (data.contracts || []) : [];
      const contractSummary = scopedContractSummary(data);
      if (!Number(contractSummary.count || 0)) return '';
      const cards = contracts.map((contract) => {
        const control = contract.control || {};
        const identity = contract.identity || {};
        const provenance = contract.provenance || {};
        const prepared = Boolean(contract.prepared || contract.lifecycle_state === 'prepared');
        const tone = contract.attention || control.active ? 'attention' : prepared ? 'prepared' : 'ready';
        const badgeLabel = contract.attention || control.active ? 'needs check' : prepared ? 'prepared' : 'aligned';
        const modelLock = (contract.expected_models || [])
          .map((model) => [model.key || model.label, model.model_id].filter(Boolean).join(': '))
          .join(' | ');
        const judges = (contract.expected_judges || [])
          .map((judge) => [judge.role, judge.model_id].filter(Boolean).join(': '))
          .join(' | ');
        const provenanceRows = [
          ['Benchmark condition', provenance.benchmark_condition_hash, 'same benchmark/prompt/sample family/judges'],
          ['Exact runset', provenance.comparison_spec_hash, 'benchmark + exact sample/run count + judges'],
          ['Benchmark spec', provenance.benchmark_spec_hash, 'prompts, scoring, rubric code'],
          ['Sample condition', provenance.sample_condition_hash, 'items/scenarios without replicate count'],
          ['Exact sample', provenance.sample_hash, 'items, sides, scenarios, runs'],
          ['Judge panel', provenance.judge_panel_hash, 'judge models and rubric'],
          ['Model batch', provenance.model_conditions_hash, 'models or served endpoints in this batch'],
          ['Batch condition', provenance.batch_condition_hash, 'benchmark condition + model batch'],
          ['Run execution', provenance.run_execution_hash, 'run id, command, output path'],
          ['Contract integrity', contract.contract_fingerprint || contract.fingerprint, 'expected work manifest'],
        ].map(([label, value, note]) => `
          <div class="hash-row">
            <span class="stat-label">${esc(label)}</span>
            <code title="${esc(value || 'unknown')}">${esc(shortHash(value))}</code>
            <span></span>
            <span class="event-detail">${esc(note)}</span>
          </div>
        `).join('');
        const compactIdentity = [
          provenance.benchmark_condition_hash ? `condition ${shortHash(provenance.benchmark_condition_hash)}` : '',
          provenance.comparison_spec_hash ? `runset ${shortHash(provenance.comparison_spec_hash)}` : '',
          provenance.model_conditions_hash ? `models ${shortHash(provenance.model_conditions_hash)}` : '',
        ].filter(Boolean).join(' · ');
        const modules = (contract.modules || []).map((module) => `
          <li>
            <strong>${esc(module.module || 'module')}</strong>
            <span class="stage-line"> ${esc(module.complete_units || 0)} / ${esc(module.expected_units || 0)} units, ${esc(module.present_artifacts || 0)} / ${esc(module.expected_artifacts || 0)} artifacts</span>
          </li>
        `).join('');
        const missing = [
          contract.missing_units ? `${contract.missing_units} incomplete expected units` : '',
          (contract.missing_required_artifacts || []).length ? `${(contract.missing_required_artifacts || []).length} missing required artifacts` : '',
          (contract.model_mismatches || []).length ? `${(contract.model_mismatches || []).length} model id mismatches` : '',
          control.active ? control.label : '',
        ].filter(Boolean).join(' | ');
        const preparedCopy = prepared
          ? `Prepared run: ${esc(contract.expected_units || 0)} planned units. Missing outputs are expected until you execute the copied command.`
          : '';
        return `
          <article class="contract-card ${tone}">
            <div class="module-top">
              <div>
                <div class="module-name">${esc(contract.run_id || 'run contract')}</div>
                <div class="stage-line">${esc(contract.lifecycle_state || contract.contract_scope || 'run_group')} | ${esc(contract.path || 'RUN_CONTRACT.json')}</div>
              </div>
              <span class="badge ${tone}">${badgeLabel}</span>
            </div>
            <div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, Number(contract.progress_percent || 0)))}%"></div></div>
            <div class="stats">
              <div><div class="stat-label">Expected</div><div class="stat-value">${esc(contract.expected_units || 0)} units</div></div>
              <div><div class="stat-label">Complete</div><div class="stat-value">${esc(contract.complete_units || 0)} units</div></div>
              <div><div class="stat-label">Artifacts</div><div class="stat-value">${esc(contract.present_artifacts || 0)} / ${esc(contract.expected_artifacts || 0)}</div></div>
              <div><div class="stat-label">Control</div><div class="stat-value">${esc(control.active ? control.label : 'clear')}</div></div>
            </div>
            ${renderContractCostEstimate(contract)}
            ${prepared ? `<div class="stage-line">${preparedCopy}</div>` : missing ? `<div class="reason">${esc(missing)}</div>` : '<div class="stage-line">Expected work matches the observed artifact footprint.</div>'}
            <details class="contract-detail">
              <summary>Model lock and judges</summary>
              <div>${esc(modelLock || 'No expected models recorded.')}</div>
              <div class="stage-line">${esc(judges ? `Judges: ${judges}` : 'No expected judges recorded.')}</div>
            </details>
            <details class="contract-detail">
              <summary>Comparable identity</summary>
              <div class="compact-hash-row">${esc(compactIdentity || 'No comparable identity hashes recorded.')}</div>
              <div class="stage-line">Family: ${esc(provenance.benchmark_family_id || identity.benchmark_family_id || 'unknown')}</div>
              <div class="hash-grid">${provenanceRows}</div>
            </details>
            ${control.present ? `<div class="score-state ${control.active ? 'blocked' : 'ready'}"><div class="stat-label">RUN_CONTROL.json</div><div>${esc(control.next_action || control.label)}</div><div class="stage-line">${esc(control.reason || '')}</div></div>` : ''}
            <details class="contract-detail">
              <summary>Expected modules</summary>
              <ul class="contract-module-list">${modules || '<li class="stage-line">No module expectations recorded.</li>'}</ul>
            </details>
            <div class="action-row">
              ${(contract.scheduler_command || contract.execute_command) ? `<button class="copy-button" type="button" data-copy="${esc(contract.scheduler_command || contract.execute_command)}">Copy scheduler command</button>` : ''}
              <button class="copy-button" type="button" data-copy="${esc(contract.path || '')}">Copy contract path</button>
              <button class="copy-button" type="button" data-copy="${esc(provenance.benchmark_condition_hash || '')}">Copy benchmark condition</button>
              <button class="copy-button" type="button" data-copy="${esc(provenance.comparison_spec_hash || '')}">Copy exact runset</button>
              <button class="copy-button" type="button" data-copy="${esc(contract.contract_fingerprint || contract.fingerprint || '')}">Copy contract hash</button>
              ${control.path ? `<button class="copy-button" type="button" data-copy="${esc(control.path)}">Copy control path</button>` : ''}
            </div>
          </article>
        `;
      }).join('');
      const detailBody = detailLoaded
        ? cards || '<div class="empty">No contract records found for this scope.</div>'
        : data.contract_loading
          ? '<div class="empty soft">Loading contract records...</div>'
          : data.contract_detail_error
            ? '<div class="empty soft">Contract records are temporarily unavailable.</div>'
            : '<div class="empty soft">Loading contract records...</div>';
      return `
        <details class="panel contract-panel" data-details-key="run-contract" ${openDetails.has('run-contract') ? 'open' : ''}>
          <summary>
            <span class="summary-copy">
              <span class="summary-title">Run Contract</span>
              <span class="panel-kicker">Collapsed by default. Expand for expected work, model locks, comparable identity hashes, artifacts, and cooperative control state.</span>
            </span>
            <span class="run-meta">
              <span class="chip">${esc(contractSummary.count || 0)} contract${Number(contractSummary.count || 0) === 1 ? '' : 's'}</span>
              <span class="chip">${esc(contractSummary.complete_units || 0)} / ${esc(contractSummary.expected_units || 0)} work units</span>
              <span class="chip">${esc(contractSummary.attention_count || 0)} gaps</span>
              <span class="chip">${esc(contractSummary.active_control_count || 0)} controls</span>
            </span>
          </summary>
          <div class="contract-grid">${detailBody}</div>
        </details>
      `;
    }

    function moduleStats(module) {
      const progress = module.progress || {};
      const cost = module.cost ? money(module.cost.total_cost_usd) : 'not tracked';
      const spend = module.spend_guard || {};
      const turns = `${progress.turn_saved || 0} / ${progress.planned_turns || 'unknown'}`;
      const conversations = `${progress.conversations_completed || 0} / ${progress.conversations_started || 0}`;
      return [
        ['Elapsed', module.elapsed || 'unknown'],
        ['Cost', cost],
        ['Spend guard', spend.label || 'not tracked'],
        ['Turns', turns],
        ['Conversations', conversations],
        ['Scores', progress.scores_saved || 0],
        ['Failures', progress.failures || 0],
        ['Incomplete', progress.conversations_incomplete || 0],
        ['Stage', module.stage || 'unknown'],
      ].map(([label, value]) => `
        <div>
          <div class="stat-label">${esc(label)}</div>
          <div class="stat-value">${esc(value)}</div>
        </div>
      `).join('');
    }

    function renderBuilderCheckbox(name, value, label, meta, checked) {
      return `
        <label class="builder-option" title="${esc(meta || label)}">
          <input type="checkbox" data-run-builder="${esc(name)}" value="${esc(value)}" ${checked ? 'checked' : ''}>
          <span>
            <span class="builder-option-name">${esc(label)}</span>
            ${meta ? `<span class="builder-option-meta">${esc(meta)}</span>` : ''}
          </span>
        </label>
      `;
    }

    function modelGroupDisplayName(name) {
      const raw = String(name || '').trim();
      if (!raw) return 'Model group';
      const words = raw.replace(/[_-]+/g, ' ').replace(/\b(\d)\s+(\d)\b/g, '$1.$2').split(/\s+/);
      const map = {
        aita: 'AITA',
        sus: 'SUS',
        gpt: 'GPT',
        glm: 'GLM',
        chatgpt: 'ChatGPT',
        xhigh: 'XHigh',
      };
      return words.map((word) => {
        const lower = word.toLowerCase();
        if (map[lower]) return map[lower];
        if (/^\d+(?:\.\d+)?$/.test(word)) return word;
        return word.slice(0, 1).toUpperCase() + word.slice(1);
      }).join(' ');
    }

    function renderModelGroupOption(group, checked) {
      const models = (group.models || []).filter(Boolean);
      const modelNames = models.map((model) => {
        const parts = modelDisplayParts(model);
        return [parts.name, parts.condition].filter(Boolean).join(' / ');
      });
      const modelSummary = modelNames.join(', ') || 'No models listed';
      const count = Number(group.count || models.length || 0);
      const countLabel = `${count} model${count === 1 ? '' : 's'}`;
      const selector = `group:${group.name}`;
      return `
        <label class="builder-option builder-model-group" title="${esc(`${selector} - ${modelSummary}`)}">
          <input type="checkbox" data-run-builder="modelGroups" value="${esc(group.name)}" ${checked ? 'checked' : ''}>
          <span>
            <span class="builder-group-main">
              <span class="builder-option-name">${esc(modelGroupDisplayName(group.name))}</span>
              <span class="builder-option-meta">${esc(countLabel)}</span>
            </span>
            <span class="builder-group-id">${esc(selector)}</span>
            <span class="builder-group-models">${esc(modelSummary)}</span>
          </span>
        </label>
      `;
    }

    function renderBuilderRadio(name, value, label, meta, checked) {
      return `
        <label class="builder-option" title="${esc(meta || label)}">
          <input type="radio" name="${esc(name)}" data-run-builder="${esc(name)}" value="${esc(value)}" ${checked ? 'checked' : ''}>
          <span>
            <span class="builder-option-name">${esc(label)}</span>
            ${meta ? `<span class="builder-option-meta">${esc(meta)}</span>` : ''}
          </span>
        </label>
      `;
    }

    function renderRunBuilder(operator) {
      const state = normalizeRunBuilderState(operator);
      const output = buildRunBuilderOutput(operator, state);
      const groups = (operator.model_groups || []).map((group) =>
        renderModelGroupOption(group, state.modelGroups.includes(group.name))
      ).join('');
      const judges = (operator.judge_sets || []).map((judge) =>
        renderBuilderCheckbox(
          'judgeSets',
          judge.name,
          judge.name,
          `${(judge.panel || []).length || 1} judge${((judge.panel || []).length || 1) === 1 ? '' : 's'}`,
          state.judgeSets.includes(judge.name),
        )
      ).join('');
      const modules = RUN_BUILDER_MODULES.map((module) =>
        renderBuilderCheckbox('modules', module.key, module.label, module.note, state.modules.includes(module.key))
      ).join('');
      const stages = RUN_BUILDER_STAGES.map((stage) =>
        renderBuilderRadio('stage', stage.key, stage.label, stage.note, state.stage === stage.key)
      ).join('');
      const sizes = RUN_BUILDER_SIZES.map((size) =>
        renderBuilderRadio('size', size.key, size.label, size.note, state.size === size.key)
      ).join('');
      return `
        <section class="run-builder" aria-label="Run Builder">
          <div class="builder-grid">
            <div class="builder-panel">
              <div class="builder-step">
                <div class="builder-step-title"><strong>1</strong> Benchmarks</div>
                <div class="panel-kicker">Choose the benchmark families first. The generated CLI loops through each selected family.</div>
                <div class="action-row builder-shortcuts" aria-label="Run builder selection shortcuts">
                  <button class="copy-button" type="button" data-run-builder-action="all-modules">All modules</button>
                </div>
                <div class="builder-options">${modules}</div>
              </div>
              <div class="builder-step">
                <div class="builder-step-title"><strong>2</strong> Model groups</div>
                <div class="panel-kicker">Model groups are presets from suite_models.yaml. Each expands to the listed model keys in the generated CLI.</div>
                <div class="action-row builder-shortcuts" aria-label="Model group shortcuts">
                  <button class="copy-button" type="button" data-run-builder-action="smoke-model-groups">Smoke groups</button>
                  <button class="copy-button" type="button" data-run-builder-action="all-model-groups">All groups</button>
                </div>
                <div class="builder-options scroll model-groups">${groups || '<div class="empty soft">No model groups configured.</div>'}</div>
              </div>
              <div class="builder-step">
                <div class="builder-step-title"><strong>3</strong> Judge panel</div>
                <div class="panel-kicker">Choose which judge set scores the generated transcripts.</div>
                <div class="action-row builder-shortcuts" aria-label="Judge set shortcuts">
                  <button class="copy-button" type="button" data-run-builder-action="all-judges">All judges</button>
                </div>
                <div class="builder-options">${judges || '<div class="empty soft">No judge sets configured.</div>'}</div>
              </div>
              <div class="builder-step">
                <div class="builder-step-title"><strong>4</strong> Stage and size</div>
                <div class="panel-kicker">Validate and prepare are safe planning stages; scheduler is the paid execution step.</div>
                <div class="builder-options">${stages}</div>
                <div class="builder-options">${sizes}</div>
              </div>
              <div class="builder-step">
                <div class="builder-step-title"><strong>5</strong> Names and limits</div>
                <div class="builder-fields">
                  <label class="builder-field">Run id prefix
                    <input type="text" data-run-builder="runIdPrefix" value="${esc(state.runIdPrefix)}" autocomplete="off" spellcheck="false">
                  </label>
                  <label class="builder-field">Output root
                    <input type="text" data-run-builder="outputRoot" value="${esc(state.outputRoot)}" autocomplete="off" spellcheck="false">
                  </label>
                  <label class="builder-field">Max calls
                    <input type="number" data-run-builder="maxActiveCalls" value="${esc(state.maxActiveCalls)}" min="1" step="1">
                  </label>
                </div>
              </div>
            </div>
            <div class="builder-panel">
              <div class="builder-output-head">
                <div>
                  <h2>Generated CLI</h2>
                  <div class="panel-kicker" id="runBuilderSummary">${esc(output.summary)}</div>
                </div>
                <div class="action-row">
                  <button class="copy-button" type="button" data-run-builder-copy="cli" data-copy="${esc(output.cli)}">Copy CLI</button>
                </div>
              </div>
              <pre class="command-pre builder-command" id="runBuilderCli">${esc(output.cli)}</pre>
              <div class="builder-output-head">
                <div>
                  <h2>Agent prompt</h2>
                  <div class="panel-kicker">For Codex or the benchmark operator skill when you want a guided run.</div>
                </div>
                <div class="action-row">
                  <button class="copy-button" type="button" data-run-builder-copy="prompt" data-copy="${esc(output.prompt)}">Copy prompt</button>
                </div>
              </div>
              <pre class="command-pre builder-prompt" id="runBuilderPrompt">${esc(output.prompt)}</pre>
            </div>
          </div>
        </section>
      `;
    }

    function renderOperator(data) {
      const operator = data.operator || {};
      if (operator.error) {
        return `<div class="empty">Could not load ${esc(operator.config_path || 'suite_models.yaml')}: ${esc(operator.error)}</div>`;
      }
      const groups = (operator.model_groups || []).map((group) => `
        <article class="inventory-row">
          <div class="inventory-title">group:${esc(group.name)}</div>
          <div class="stage-line">${esc(group.count)} model${group.count === 1 ? '' : 's'} - ${esc((group.models || []).join(', '))}</div>
          <button class="copy-button" type="button" data-copy="${esc(`group:${group.name}`)}">Copy selector</button>
        </article>
      `).join('');
      const judges = (operator.judge_sets || []).map((judge) => `
        <article class="inventory-row">
          <div class="inventory-title">--judge-set ${esc(judge.name)}</div>
          <div class="stage-line">${esc(judge.description || '')}</div>
          <div class="event-detail">${esc((judge.panel || []).join(' | '))}</div>
        </article>
      `).join('');
      const models = (operator.models || []).slice(0, 24).map((model) => `
        <article class="inventory-row">
          <div class="inventory-title">${esc(model.key)} <span class="chip">${esc(model.endpoint)}</span></div>
          <div class="stage-line">${esc(model.label || '')}</div>
          <div class="event-detail">${esc(model.model_id || '')}</div>
        </article>
      `).join('');
      const commands = (operator.commands || []).map((command) => `
        <article class="command-card">
          <div class="command-title">${esc(command.title)}</div>
          <div class="stage-line">${esc(command.description)}</div>
          <pre class="command-pre">${esc(command.command)}</pre>
          <button class="copy-button" type="button" data-copy="${esc(command.command)}">Copy command</button>
        </article>
      `).join('');
      return `
        <details class="operator-section operator-tools command-center" data-details-key="operator-run-builder" ${openDetails.has('operator-run-builder') ? 'open' : ''}>
          <summary class="section-head">
            <div>
              <h2>Operator tools</h2>
              <div class="panel-kicker">Run builder, model selectors, judge sets, and CLI helpers. Copy-only; this drawer does not read old ledgers or launch paid calls.</div>
            </div>
            <div class="run-meta">
              <span class="chip">run builder</span>
              <span class="chip">${esc((operator.model_groups || []).length)} groups</span>
              <span class="chip">${esc((operator.models || []).length)} models</span>
              <span class="chip">${esc((operator.judge_sets || []).length)} judge sets</span>
            </div>
          </summary>
          <div class="operator-body">
            ${renderRunBuilder(operator)}
            <details class="advanced-cli">
              <summary>
                <h2>Advanced CLI</h2>
                <div class="panel-kicker">Raw selectors, judge panels, model inventory, and legacy command cards.</div>
              </summary>
              <div class="advanced-cli-body">
                <div class="inventory-block">
                  <div>
                    <h2>Selectors</h2>
                    <div class="panel-kicker">Use these with suite_tools.model_config --models.</div>
                  </div>
                  <div class="model-list">${groups || '<div class="empty">No model groups configured.</div>'}</div>
                  <div>
                    <h2>Judge Sets</h2>
                    <div class="panel-kicker">Use these with --judge-set.</div>
                  </div>
                  <div class="model-list">${judges || '<div class="empty">No judge sets configured.</div>'}</div>
                  <details>
                    <summary>Configured models</summary>
                    <div class="model-list">${models || '<div class="empty">No models configured.</div>'}</div>
                  </details>
                </div>
                <div class="inventory-block">
                  <div>
                    <h2>Raw command cards</h2>
                    <div class="panel-kicker">Low-level commands retained for debugging and manual operator work.</div>
                  </div>
                  <div class="command-list">${commands || '<div class="empty">No commands configured.</div>'}</div>
                </div>
              </div>
            </details>
          </div>
        </details>
      `;
    }

    function renderAttention(data) {
      const modules = workingRawModules(data)
        .filter((module) => module.attention)
        .filter((module) => showAcknowledged || !isHiddenModule(module))
        .sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')));
      const hiddenCount = workingRawModules(data).filter((module) => (module.attention || isRejectedModule(module)) && isHiddenModule(module)).length;
      if (!modules.length) {
        return hiddenCount
          ? '<div class="empty">No active attention items. Acknowledged diagnostics are hidden from this working view.</div>'
          : '<div class="empty">No failed or incomplete modules in the selected run scope.</div>';
      }
      return modules.slice(0, 8).map((module) => {
        const attention = module.attention || {};
        const ackKey = moduleAckKey(module);
        const acknowledgedModule = isAcknowledgedModule(module);
        const rejectedModule = isRejectedModule(module);
        const examples = (attention.incomplete_examples || []).map((example) => `<li>${esc(example)}</li>`).join('');
        const remainder = attention.incomplete_count > (attention.incomplete_examples || []).length
          ? `<li>${esc(attention.incomplete_count - attention.incomplete_examples.length)} more incomplete conversations</li>`
          : '';
        const state = module.score_state || {};
        const nextAction = attention.action || state.action || 'Inspect the run ledger directly.';
        return `
          <article class="attention-item ${(acknowledgedModule || rejectedModule) ? 'acknowledged' : ''}">
            <div class="attention-top">
              <div>
                <div class="attention-name">${esc(module.run_id)} / ${esc(module.module_path || module.module)}</div>
                <div class="stage-line">${esc(attention.title || 'Needs review')} | updated ${esc(formatTime(module.updated_at))}</div>
              </div>
              ${statusBadge(module)}
            </div>
            ${failureClassificationMarkup(attention)}
            <div class="failure-detail">
              <div class="stat-label">Recorded failure</div>
              <div>${esc(attention.reason || 'No failure detail recorded.')}</div>
            </div>
            <div class="score-state ${esc(state.kind || '')}">
              <div class="stat-label">Next action</div>
              <div>${esc(nextAction)}</div>
            </div>
            ${(examples || remainder) ? `<ul class="example-list">${examples}${remainder}</ul>` : ''}
            <div class="action-row">
              <button class="copy-button" type="button" data-copy="${esc(triageCopy(module))}">Copy triage commands</button>
              <button class="copy-button" type="button" data-copy="${esc(failureCopy(module))}">Copy failure</button>
              <button class="copy-button" type="button" data-copy="${esc(module.status_path || '')}">Copy status path</button>
              <button class="copy-button" type="button" data-copy="${esc(module.events_path || '')}">Copy events path</button>
              ${module.status_path ? `<button class="copy-button danger" type="button" data-disposition="${esc(module.status_path)}" data-disposition-action="reject">Reject from analysis</button>` : ''}
              ${ackKey ? `<button class="copy-button" type="button" data-ack="${esc(ackKey)}">${acknowledgedModule ? 'Unhide' : 'Hide locally'}</button>` : ''}
            </div>
          </article>
        `;
      }).join('');
    }

    function renderModule(module) {
      const acknowledgedModule = isAcknowledgedModule(module);
      const rejectedModule = isRejectedModule(module);
      const ackKey = moduleAckKey(module);
      const progress = module.progress || {};
      const state = module.score_state || {};
      const percent = progress.percent;
      const width = percent == null ? 0 : Math.max(0, Math.min(100, Number(percent)));
      const latest = module.latest_event ? `${eventTitle(module.latest_event)} | ${relativeTime(module.latest_event.timestamp)}` : 'No events yet';
      const detailsKey = moduleKey(module);
      return `
        <article class="module ${(acknowledgedModule || rejectedModule) ? 'acknowledged' : ''}" data-severity="${esc(module.severity)}">
          <div class="module-top">
            <div>
              <div class="module-name">${esc(module.module_path || module.module)}</div>
              <div class="stage-line">${esc(latest)}</div>
            </div>
            ${statusBadge(module)}
          </div>
          <div class="bar"><div class="fill" style="width:${width}%"></div></div>
          <div class="stage-line">${percent == null ? 'Progress is event-based' : `Progress ${percent}%`}</div>
          <div class="stats">${moduleStats(module)}</div>
          <div class="score-state ${esc(state.kind || '')}">
            <div class="stat-label">Score state</div>
            <div><strong>${esc(state.label || 'Unknown')}</strong> - ${esc(state.detail || 'Inspect the run ledger directly.')}</div>
            <div class="stage-line">${esc(state.action || '')}</div>
          </div>
          ${module.attention ? `<div class="reason">${esc(module.attention.title)}: ${esc(module.attention.reason)}</div>` : ''}
          <div class="action-row">
            <button class="copy-button" type="button" data-copy="${esc(module.output_dir || '')}">Copy output path</button>
            ${module.attention ? `<button class="copy-button" type="button" data-copy="${esc(failureCopy(module))}">Copy failure</button>` : ''}
            ${module.attention && module.status_path ? `<button class="copy-button danger" type="button" data-disposition="${esc(module.status_path)}" data-disposition-action="reject">Reject from analysis</button>` : ''}
            ${rejectedModule && module.status_path ? `<button class="copy-button" type="button" data-disposition="${esc(module.status_path)}" data-disposition-action="restore">Restore diagnostic</button>` : ''}
            ${module.attention && ackKey ? `<button class="copy-button" type="button" data-ack="${esc(ackKey)}">${acknowledgedModule ? 'Unhide' : 'Hide locally'}</button>` : ''}
          </div>
          <details data-details-key="${esc(detailsKey)}" ${openDetails.has(detailsKey) ? 'open' : ''}>
            <summary>Artifacts and events</summary>
            <div class="path-block">${esc(module.output_dir)}<br>${esc(module.status_path)}<br>${esc(module.events_path)}</div>
            <ul class="mini-events">
              ${(module.recent_events || []).slice().reverse().map((event) => `<li class="mini-event">${esc([event.timestamp, event.event, event.model, event.item_idx != null ? `item ${event.item_idx}` : '', event.side, event.failure_reason].filter(Boolean).join(' | '))}</li>`).join('') || '<li class="mini-event">No events yet.</li>'}
            </ul>
          </details>
        </article>
      `;
    }

    function renderGroups(data) {
      const groups = (data.groups || [])
        .filter((group) => runScopeMatches({group: group.run_id, run_id: group.run_id}, data))
        .map((group) => {
        const baseModules = showAcknowledged
          ? (group.modules || [])
          : (group.modules || []).filter((module) => !isHiddenModule({...module, run_id: group.run_id}));
        const modules = activeFilter === 'all'
          ? baseModules
          : baseModules.filter((module) => module.severity === activeFilter);
        return {...group, modules};
      }).filter((group) => (group.modules || []).length);
      if (!groups.length) return '<div class="empty">No modules match the current filter.</div>';
      return groups.map((group) => `
        <section class="run-group">
          <div class="run-head">
            <div>
              <div class="run-title">${esc(group.run_id)}</div>
              <div class="stage-line">Elapsed ${esc(group.elapsed)} | updated ${esc(formatTime(group.updated_at))} | latest event ${esc(relativeTime(group.latest_event_at))}</div>
            </div>
            <div class="run-meta">
              <span class="chip">${esc(group.modules.length)} shown</span>
              <span class="chip">${esc(group.attention_count || 0)} attention</span>
              <span class="chip">${esc(group.ready_count || 0)} scored</span>
              <span class="chip">${money(group.cost_total_usd)} tracked</span>
            </div>
          </div>
          <div class="module-grid">${group.modules.map(renderModule).join('')}</div>
        </section>
      `).join('');
    }

	    function renderFamilyPanel(data) {
	      const family = familyForScope(evidenceRunScope, data);
	      if (!family) return '';
	      const memberIds = (family.member_run_ids || []).map(String);
	      const byId = new Map((data.groups || []).map((group) => [String(group.run_id || ''), group]));
	      let genDone = 0;
	      let genTotal = 0;
	      let running = 0;
	      let errors = 0;
	      let cost = 0;
	      const rows = memberIds.map((runId) => {
	        const group = byId.get(runId) || null;
	        const contractSummary = (group || {}).contract_summary || {};
	        const done = Number(contractSummary.complete_units || 0);
	        const total = Number(contractSummary.expected_units || 0);
	        const runCount = Number((group || {}).running_count || 0);
	        const moduleList = (group || {}).modules || [];
	        const errCount = moduleList.filter((module) => String(module.status || '').startsWith('failed')).length;
	        const groupCost = Number((group || {}).cost_total_usd || 0);
	        genDone += done;
	        genTotal += total;
	        running += runCount;
	        errors += errCount;
	        cost += groupCost;
	        const severity = (group || {}).severity || (group ? 'idle' : 'prepared');
	        const stateNote = group ? '' : ' · contract only';
	        return `
	          <div class="family-member">
	            <div class="family-member-id">
	              <span class="chip ${stageClass(severity)}">${esc(severity)}</span>
	              <strong>${esc(shortHash(runId))}</strong>
	              <span class="panel-kicker">${esc(runId)}${stateNote}</span>
	            </div>
	            <div class="family-member-metrics">
	              <span class="chip">${esc(done)}/${esc(total)} gen</span>
	              <span class="chip ${runCount ? 'running' : ''}">${esc(runCount)} running</span>
	              <span class="chip ${errCount ? 'attention' : ''}">${esc(errCount)} err</span>
	              <span class="chip">${money(groupCost)}</span>
	              <button class="copy-button" type="button" data-scope-select="${esc(runId)}">View run</button>
	            </div>
	          </div>`;
	      }).join('');
	      return `
	        <section class="panel family-panel" aria-label="Logical run family">
	          <div class="panel-head">
	            <div>
	              <h2>Run family · ${esc(shortHash(family.prereg_sha256))}</h2>
	              <div class="panel-kicker">${esc(memberIds.length)} run dirs share one prereg freeze. Counters aggregate across members.</div>
	            </div>
	            <div class="run-meta">
	              <span class="chip">${esc(genDone)}/${esc(genTotal)} gen</span>
	              <span class="chip ${running ? 'running' : ''}">${esc(running)} running</span>
	              <span class="chip ${errors ? 'attention' : ''}">${esc(errors)} errors</span>
	              <span class="chip">${money(cost)} tracked</span>
	            </div>
	          </div>
	          <div class="family-members">${rows}</div>
	        </section>`;
	    }

	    function render(data) {
	      syncEvidenceDefaultsForRun(data);
	      lastData = data;
	      const isFirstRender = firstPaint;
	      const feedState = captureEvidenceFeedState();
	      const nextEvidenceViewSignature = evidenceViewSignature(data);
	      const evidenceChanged = !lastEvidenceViewSignature || nextEvidenceViewSignature !== lastEvidenceViewSignature;
	      if (isFirstRender && !pendingFeedPanelScroll && !evidenceAutoFollow) feedState.pageY = 0;
	      updateModelRegistry(data);
	      rememberFresh(data.latest_events || [], data.evidence_feed || [], Boolean(data.evidence_detail_key));
	      lastRefresh.textContent = `Dashboard refreshed ${formatTime(data.generated_at)} from ${data.results_root}`;
	      liveLabel.textContent = freshEvents.size ? `${freshEvents.size} new ledger event${freshEvents.size === 1 ? '' : 's'}` : 'Watching ledgers';
      const apiNotice = data.error
        ? `<div class="empty soft">Dashboard API is waiting on readable ledgers. Showing a safe empty snapshot until the next successful refresh. ${esc(data.error)}</div>`
        : '';
	      if (!data.groups.length) {
	        updateTopSummary(data, 0);
	        syncTopScopeControl(data);
	        const emptyHtml = `${apiNotice}<div class="empty">No run ledgers found yet. Waiting for prepared contracts, RUN_STATUS.json, or RUN_EVENTS.jsonl.</div>`;
	        if (emptyHtml !== lastAppHtml) {
	          replaceAppHtml(emptyHtml);
	        }
	        syncPersistentControls();
	        return;
	      }
      const summary = data.summary || {};
	      const contractStats = scopedContractSummary(data);
	      const visibleModules = workingModules(data);
	      const visibleAttention = visibleModules.filter((module) => module.attention || module.severity === 'attention').length;
	      const rejectedCount = workingRawModules(data).filter(isRejectedModule).length;
	      const hiddenAttention = workingRawModules(data).filter((module) => (module.attention || isRejectedModule(module)) && isHiddenModule(module)).length;
	      const scopedEvents = scopedLatestEvents(data);
	      const visibleLanes = visibleFlowLanes(data);
	      const scopedGroups = (data.groups || []).filter((group) => runScopeMatches({group: group.run_id, run_id: group.run_id}, data));
	      const scopedCost = scopedCostTotal(data);
	      const scopedFailures = evidenceRunScope === 'all' ? Number(summary.failed_count || 0) : scopedFailedCount(data);
	      const scopedSpendIssues = evidenceRunScope === 'all' ? Number(summary.spend_attention_count || 0) : scopedSpendIssueCount(data);
	      updateTopSummary(data, visibleAttention);
	      syncTopScopeControl(data);
	      const html = `
	        ${apiNotice}
	        ${renderHudRail(data, visibleAttention)}
	        ${renderCreditAlert(data)}
	        ${renderFamilyPanel(data)}
	        <section class="mission-grid" aria-label="Run control mission surface">
	          ${renderWorkQueue(data)}
	          ${renderEvidenceFeed(data)}
	          ${renderRadar(data)}
	        </section>
	        ${renderContracts(data)}
	        <section class="workbench">
	          <section class="panel">
	            <div class="panel-head">
	            <div>
              <h2>Attention Queue</h2>
              <div class="panel-kicker">Triage lane. Hide acknowledged diagnostics from this working view without changing run ledgers or evidence.</div>
            </div>
            <div class="run-meta">
              <span class="chip">${esc(visibleAttention)} active</span>
              <span class="chip">${esc(hiddenAttention)} hidden</span>
              <button class="copy-button" type="button" data-toggle-ack>${showAcknowledged ? 'Hide rejected/hidden' : 'Show rejected/hidden'}</button>
            </div>
          </div>
            <div class="attention-list">${renderAttention(data)}</div>
          </section>
          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Latest Events</h2>
                <div class="panel-kicker">Newest ledger writes in ${esc(runScopeLabel(data))}.</div>
              </div>
              <span class="chip">${esc(scopedEvents.length)} shown</span>
            </div>
            <div class="event-stream">${scopedEvents.slice(0, 30).map(renderEvent).join('') || '<div class="empty">No run events found for the selected scope.</div>'}</div>
	          </section>
	        </section>
	        <details class="panel advanced-inspection" data-details-key="advanced-inspection" ${openDetails.has('advanced-inspection') ? 'open' : ''}>
	          <summary class="panel-head">
	            <div>
	              <h2>Advanced inspection</h2>
	              <div class="panel-kicker">Scoped run flow, diagnostics, artifact snapshots, and raw ledger groups.</div>
	            </div>
	            <div class="run-meta">
	              <span class="chip">${esc(runScopeLabel(data))}</span>
	              <span class="chip">${esc(visibleLanes.length)} lanes</span>
	              <span class="chip">${esc(scopedGroups.length)} raw group${scopedGroups.length === 1 ? '' : 's'}</span>
		              <span class="chip">${money(scopedCost)} tracked</span>
	            </div>
	          </summary>
	          <div class="advanced-inspection-body">
	            <details class="advanced-subpanel" data-details-key="advanced-flow" ${openDetails.has('advanced-flow') ? 'open' : ''}>
	              <summary>
	                <span class="advanced-subpanel-title">
	                  <strong>Run Flow</strong>
	                  <span class="panel-kicker">Lane board for Prepared, Queued, Generating, Needs Scoring, Scoring, Scored, and Attention.</span>
	                </span>
	                <span class="run-meta">
	                  <span class="chip">${esc(visibleLanes.length)} lanes</span>
	                  <span class="chip">${esc(visibleLanes.find((lane) => lane.id === 'prepared')?.work_group_count || 0)} prepared groups</span>
	                  <span class="chip">${esc(visibleLanes.find((lane) => lane.id === 'queued')?.work_group_count || 0)} queued groups</span>
	                  <span class="chip">${esc((visibleLanes.find((lane) => lane.id === 'generating')?.work_group_count || 0) + (visibleLanes.find((lane) => lane.id === 'scoring')?.work_group_count || 0))} active</span>
	                  <span class="chip">${esc(visibleLanes.find((lane) => lane.id === 'attention')?.work_group_count || 0)} attention</span>
	                </span>
	              </summary>
	              <div class="advanced-subpanel-content">${renderRunFlow(data)}</div>
	            </details>
	            <details class="advanced-subpanel" data-details-key="advanced-diagnostics" ${openDetails.has('advanced-diagnostics') ? 'open' : ''}>
	              <summary>
	                <span class="advanced-subpanel-title">
	                  <strong>Diagnostics Summary</strong>
	                  <span class="panel-kicker">Scheduler, spend, contract gaps, and loaded-ledger counters.</span>
	                </span>
	                <span class="run-meta">
	                  <span class="chip">${esc(visibleAttention)} attention</span>
	                  <span class="chip">${esc(contractStats.attention_count || 0)} gaps</span>
		                  <span class="chip">${money(scopedCost)}</span>
	                </span>
	              </summary>
	              <div class="advanced-subpanel-content">
	                <section class="summary-grid" aria-label="Benchmark run diagnostics summary">
		                  ${metric('Attention', visibleAttention, hiddenAttention ? `${hiddenAttention} rejected/hidden` : `${scopedFailures} failed modules`, visibleAttention ? 'attention' : '')}
	                  ${metric('Running', visibleModules.filter((module) => module.severity === 'running').length, 'active modules', 'running')}
	                  ${metric('Paid calls', `${summary.paid_call_active_count || 0} / ${summary.paid_call_max_active || 0}`, 'global active leases', summary.paid_call_active_count ? 'running' : '')}
	                  ${metric('Rate-limit pause', summary.paid_call_rate_limit_cooldown_count || 0, summary.paid_call_next_cooldown_seconds == null ? 'no active cooldowns' : `next release ${duration(summary.paid_call_next_cooldown_seconds)}`, summary.paid_call_rate_limit_cooldown_count ? 'attention' : '')}
	                  ${metric('Queued', summary.scheduler_queued_count || ((data.flow || {}).counts || {}).queued || 0, `${summary.scheduler_count || 0} scheduled runs`)}
	                  ${metric('Active elapsed', summary.active_elapsed || 'none', 'longest running module')}
	                  ${metric('Loaded elapsed', summary.suite_elapsed || 'none', 'longest loaded group')}
	                  ${metric('Scheduler ETA', duration(summary.scheduler_eta_seconds), summary.scheduler_eta_seconds == null ? 'pending completed units' : 'nearest active scheduled run')}
	                  ${metric('Scored', summary.score_ready_count || 0, `${summary.ready_count || 0} scored modules`, 'ready')}
		                  ${metric('Tracked cost', money(scopedCost), `${visibleModules.length} modules`)}
		                  ${metric('Spend guard', scopedSpendIssues, 'modules over budget or low credit', scopedSpendIssues ? 'attention' : '')}
	                  ${metric('Latest event', relativeTime(summary.latest_event_at), formatTime(summary.latest_event_at))}
	                  ${metric('Contract gaps', contractStats.attention_count || 0, `${contractStats.complete_units || 0} / ${contractStats.expected_units || 0} expected units`, contractStats.attention_count ? 'attention' : '')}
	                  ${metric('Rejected', rejectedCount || summary.rejected_count || 0, 'excluded diagnostics')}
	                </section>
	              </div>
	            </details>
	            <details class="advanced-subpanel" data-details-key="advanced-snapshots" ${openDetails.has('advanced-snapshots') ? 'open' : ''}>
	              <summary>
	                <span class="advanced-subpanel-title">
	                  <strong>Module Snapshots</strong>
	                  <span class="panel-kicker">Artifact cards with progress counters and latest saved turn context.</span>
	                </span>
	                <span class="run-meta">
	                  <span class="chip">${esc(visibleModules.length)} modules</span>
	                  <span class="chip">${esc(visibleModules.filter((module) => module.severity === 'running').length)} active</span>
	                </span>
	              </summary>
	              <div class="advanced-subpanel-content">
	                <div class="activity-grid">${renderLiveActivity(data)}</div>
	              </div>
	            </details>
	            <details class="advanced-subpanel" data-details-key="advanced-raw-groups" ${openDetails.has('advanced-raw-groups') ? 'open' : ''}>
	              <summary>
	                <span class="advanced-subpanel-title">
	                  <strong>Raw Run Groups</strong>
	                  <span class="panel-kicker">Raw ledger groups under ${esc(data.results_root)} for artifact-level debugging.</span>
	                </span>
	                <span class="run-meta">
	                  <span class="chip">${esc(scopedGroups.length)} group${scopedGroups.length === 1 ? '' : 's'}</span>
	                  <span class="chip">${esc(runScopeLabel(data))}</span>
	                </span>
	              </summary>
	              <div class="advanced-subpanel-content">
	                <div class="toolbar section-toolbar">
	                  ${filters.map(([key, label]) => `<button class="filter-button" type="button" data-filter="${key}" aria-pressed="${activeFilter === key}">${label}</button>`).join('')}
	                </div>
	                <div class="run-list">${renderGroups(data)}</div>
	              </div>
	            </details>
	          </div>
	        </details>
	        ${renderOperator(data)}
	      `;
	      // Repaint once per render, and only when the markup actually changed.
	      // A wholesale innerHTML swap tears down and rebuilds the DOM (the
	      // flash-to-blank-then-numbers); skipping identical paints keeps the
	      // surface steady across the multiple render() calls each poll makes.
	      const htmlChanged = html !== lastAppHtml;
	      if (htmlChanged) {
	        replaceAppHtml(html);
	      }
	      syncPersistentControls();
	      drawEvidenceTrace(data);
	      if (htmlChanged) {
	        bindEvidenceTrace();
	        syncEvidenceFeedScroll(feedState, {evidenceChanged});
	      }
	      lastEvidenceViewSignature = nextEvidenceViewSignature;
	      window.requestAnimationFrame(() => {
	        drawEvidenceTrace(data);
	        if (htmlChanged) bindEvidenceTrace();
	      });
	    }

    document.addEventListener('click', (event) => {
      const scopeSelect = event.target.closest('[data-scope-select]');
      if (scopeSelect) {
        applyScopeSelection(scopeSelect.dataset.scopeSelect || 'latest');
        return;
      }

      const copyButton = event.target.closest('[data-copy]');
      if (copyButton) {
        const value = copyButton.dataset.copy || '';
        copyValue(copyButton, value);
        return;
      }

      const builderAction = event.target.closest('[data-run-builder-action]');
      if (builderAction) {
        handleRunBuilderAction(builderAction);
        return;
      }

      const ackButton = event.target.closest('[data-ack]');
      if (ackButton) {
        const key = ackButton.dataset.ack || '';
        if (key) {
          if (acknowledged.has(key)) acknowledged.delete(key);
          else acknowledged.add(key);
          saveAcknowledged();
          if (lastData) render(lastData);
        }
        return;
      }

      const dispositionButton = event.target.closest('[data-disposition]');
      if (dispositionButton) {
        writeDisposition(
          dispositionButton,
          dispositionButton.dataset.disposition || '',
          dispositionButton.dataset.dispositionAction || 'reject',
        );
        return;
      }

	      const toggleAck = event.target.closest('[data-toggle-ack]');
	      if (toggleAck) {
	        showAcknowledged = !showAcknowledged;
	        if (lastData) render(lastData);
	        return;
	      }

	      const quickStage = event.target.closest('[data-quick-stage]');
	      if (quickStage) {
	        applyStageFilter(quickStage.dataset.quickStage || 'all', true);
	        return;
	      }

	      const evidenceWindow = event.target.closest('[data-evidence-window]');
	      if (evidenceWindow) {
	        setEvidenceTraceWindow(evidenceWindow.dataset.evidenceWindow || 'all');
	        return;
	      }

	      const evidenceContent = event.target.closest('[data-evidence-content]');
	      if (evidenceContent) {
	        setEvidenceContentFilter(evidenceContent.dataset.evidenceContent || 'all');
	        return;
	      }

		      const quickAction = event.target.closest('[data-quick-action]');
		      if (quickAction) {
		        const action = quickAction.dataset.quickAction || '';
		        if (action === 'follow-live') {
		          setEvidenceFollow(!evidenceAutoFollow);
		        }
		        return;
		      }

		      const queueToggle = event.target.closest('[data-queue-toggle]');
		      if (queueToggle) {
		        const key = queueToggle.dataset.queueToggle || '';
		        if (key) {
		          queueExpansionState.set(key, queueToggle.getAttribute('aria-expanded') !== 'true');
		          if (lastData) render(lastData);
		        }
		        return;
		      }

		      const stageFilter = event.target.closest('[data-stage-filter]');
		      if (stageFilter) {
		        applyStageFilter(stageFilter.dataset.stageFilter || 'all');
		        return;
		      }

	      const jumpLive = event.target.closest('[data-jump-live]');
	      if (jumpLive) {
	        setEvidenceFollow(true);
	        return;
	      }

	      const evidenceCard = event.target.closest('.feed-card[data-evidence-key]');
	      if (evidenceCard) {
	        selectEvidenceTraceKey(evidenceCard.dataset.evidenceKey || '');
	        return;
	      }

	      const summary = event.target.closest('summary');
      if (summary && summary.parentElement && summary.parentElement.matches('details[data-details-key]')) {
        window.setTimeout(() => {
          const details = summary.parentElement;
          const key = details.dataset.detailsKey;
          if (!key) return;
	          if (details.open) {
	            openDetails.add(key);
	            if (key === 'run-contract') {
	              ensureContractDetails(lastData, {renderOnComplete: false}).then(() => {
	                if (lastData) render(lastData);
	              });
	            }
	          } else {
            openDetails.delete(key);
          }
        }, 0);
        return;
      }

      const button = event.target.closest('[data-filter]');
      if (!button) return;
      activeFilter = button.dataset.filter || 'all';
      if (lastData) render(lastData);
    });

	    document.addEventListener('keydown', (event) => {
	      const evidenceCard = event.target.closest('.feed-card[data-evidence-key]');
	      if (!evidenceCard || !['Enter', ' '].includes(event.key)) return;
	      event.preventDefault();
	      selectEvidenceTraceKey(evidenceCard.dataset.evidenceKey || '');
	    });

	    function applyScopeSelection(value) {
	      evidenceRunScope = value || 'latest';
	      suppressFreshOnNextRender = true;
	      runsEtag = '';
	      resetEvidenceFiltersToAll();
	      lastEvidenceRunFingerprint = evidenceRunFingerprint(lastData);
	      evidenceAutoFollow = false;
	      evidenceTraceAutoFollow = false;
	      pendingFeedPanelScroll = true;
	      liveLabel.textContent = 'Loading selected run';
	      if (refreshInFlight) scopeRefreshPending = true;
	      else refresh();
	    }

	    document.addEventListener('change', (event) => {
	      const evidenceScope = event.target.closest('[data-evidence-run-scope]');
	      if (evidenceScope) {
	        applyScopeSelection(evidenceScope.value || 'latest');
	        return;
	      }

	      const control = event.target.closest('[data-run-builder]');
	      if (!control) return;
	      const root = control.closest('.run-builder');
      readRunBuilderState(root);
      updateRunBuilderOutput();
    });

    document.addEventListener('input', (event) => {
      const control = event.target.closest('[data-run-builder]');
      if (!control) return;
      const root = control.closest('.run-builder');
      readRunBuilderState(root);
      updateRunBuilderOutput();
    });

    closeCopyPanel.addEventListener('click', () => {
      copyPanel.hidden = true;
    });

    selectCopyText.addEventListener('click', () => {
      copyTextarea.focus();
      copyTextarea.select();
    });

    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const current = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
        setTheme(current === 'dark' ? 'light' : 'dark', true);
      });
    }
    window.addEventListener('resize', () => drawEvidenceTrace(lastData));

    if (window.matchMedia) {
      try {
        const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
        colorSchemeQuery.addEventListener('change', (event) => {
          if (!storedTheme()) setTheme(event.matches ? 'dark' : 'light', false);
        });
      } catch (error) {}
    }

    async function refresh() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      const controller = new AbortController();
      const requestTimeout = window.setTimeout(
        () => controller.abort(),
        Math.max(5000, pollMs * 2),
      );
	      try {
	        const headers = runsEtag ? {'If-None-Match': runsEtag} : {};
	        const runsUrl = detailUrl('/api/runs', {scope: evidenceRunScope});
	        const response = await fetch(runsUrl, {
          cache: 'no-cache',
          headers,
          signal: controller.signal,
        });
        if (response.status === 304) {
          noteStaticSnapshot();
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        runsEtag = response.headers.get('ETag') || '';
        const data = await response.json();
        carryForwardDetail(lastData, data);
        await hydrateDashboardDetails(data);
        render(data);
      } catch (error) {
        console.warn('Dashboard refresh paused', error);
        liveLabel.textContent = lastData ? 'Waiting for dashboard API' : 'Waiting for ledgers';
        if (lastData) {
          lastRefresh.textContent = `Dashboard refresh paused; showing last snapshot from ${formatTime(lastData.generated_at)}`;
          return;
        }
        replaceAppHtml('<div class="empty">Waiting for dashboard data. Start or refresh the dashboard server to load run ledgers.</div>');
	      } finally {
        window.clearTimeout(requestTimeout);
	        refreshInFlight = false;
	        window.clearTimeout(refreshTimer);
	        const nextRefreshDelay = scopeRefreshPending ? 0 : pollMs;
	        scopeRefreshPending = false;
	        refreshTimer = window.setTimeout(refresh, nextRefreshDelay);
      }
    }
    refresh();
