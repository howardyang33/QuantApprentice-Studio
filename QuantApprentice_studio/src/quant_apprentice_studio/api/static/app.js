const state = {
  profile: "gpt_oss_20b_final",
  overview: null,
  liveConfig: null,
  lessons: [],
  markets: [],
  importedTeachers: [],
  teacherLibraries: [],
  activeTeacherLibraryId: "",
  importedLessonDetail: null,
  projectView: null,
  runSpecs: [],
  activeRun: null,
  latestRunSpec: null,
  runMonitor: null,
  teacherZoo: null,
  lessonSet: null,
  provenance: null,
  taskIntake: null,
  datasetManifest: null,
  klineJob: null,
  klineJobPoller: null,
  wizardBundle: null,
  chatSession: null,
  taskState: null,
  chatMessages: [],
  chatArtifactLinks: [],
  chatRecommendedActions: [],
  simpleKlinePoller: null,
  simpleWorkflowPoller: null,
  simpleWorkflowPollErrors: 0,
  simpleRunStatus: null,
  simpleSelectedTimelineNodeId: "",
  simpleLastWorkflowSignature: "",
  simpleScoringResult: null,
  latestScoringProvenance: null,
  simpleTyping: false,
  simpleAttachedFilename: "",
  expertChatMessages: [],
  expertTyping: false,
  expertAttachedFilename: "",
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function formatDateYmd(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}${month}${day}`;
}

function defaultSimpleKlineEarliestDate() {
  // 120 trading days is roughly 170-180 calendar days. Keep Simple Mode lightweight.
  const date = new Date();
  date.setDate(date.getDate() - 180);
  return formatDateYmd(date);
}

function setSimpleKlineDefaults() {
  const input = $("simple-kline-earliest");
  if (!input) return;
  if (!input.value || input.value === "20190101") {
    input.value = defaultSimpleKlineEarliestDate();
  }
}

function setWorkspace(name) {
  const aliases = {
    library: "teacher-zoo",
    provenance: "audit-trail",
    overview: "dataset-lab",
  };
  const targetName = aliases[name] || name;
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.workspaceTarget === targetName);
  });
  document.querySelectorAll(".workspace-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.workspace === targetName);
  });
  document.body.classList.toggle("simple-focus", targetName === "simple");
  document.body.classList.toggle("expert-focus", targetName !== "simple");
  if ($("mode-simple-toggle")) $("mode-simple-toggle").classList.toggle("is-active", targetName === "simple");
  if ($("mode-expert-toggle")) $("mode-expert-toggle").classList.toggle("is-active", targetName !== "simple");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function wireWorkspaceTabs() {
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.addEventListener("click", () => setWorkspace(button.dataset.workspaceTarget));
  });
}

function sourcePill(source) {
  if (!source) return '<span class="pill">unknown</span>';
  if (["current_workflow_asset", "user_trained", "explicit_final_lesson_state_json"].includes(source)) return '<span class="pill accent">本次训练老师库</span>';
  if (["imported_final_asset", "imported_frozen_teacher", "imported_frozen_teacher_zoo", "built_in_baseline", "lesson_alias"].includes(source)) return '<span class="pill">A股趋势回调与突破教师库</span>';
  if (["demo_asset", "imported_demo"].includes(source)) return '<span class="pill warn">开发测试资产</span>';
  if (source === "mixed_asset") return '<span class="pill warn">混合来源</span>';
  if (source === "external_asset") return '<span class="pill">外部数据</span>';
  return `<span class="pill">${escapeHtml(source)}</span>`;
}

function activeTeacherLibrary() {
  const libraries = state.teacherLibraries || [];
  const activeId = state.activeTeacherLibraryId || "";
  return libraries.find((item) => item.teacher_library_id === activeId)
    || libraries.find((item) => item.default_for_market)
    || libraries[0]
    || null;
}

function teacherLibraryName(id = "") {
  const libraries = state.teacherLibraries || [];
  const item = libraries.find((row) => row.teacher_library_id === id) || activeTeacherLibrary();
  return item?.display_name_zh || "A股趋势回调与突破教师库";
}

function sourceLabelZh(source = "") {
  const value = String(source || "").trim();
  if (["current_workflow_asset", "user_trained", "explicit_final_lesson_state_json"].includes(value)) return "本次训练老师库";
  if (["imported_final_asset", "imported_frozen_teacher", "imported_frozen_teacher_zoo", "built_in_baseline", "lesson_alias"].includes(value)) return teacherLibraryName();
  if (["demo_asset", "imported_demo"].includes(value)) return "开发测试资产";
  if (value === "unavailable" || value === "unresolved" || !value) return "未生成";
  return value;
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function summaryCards(containerId, rows) {
  $(containerId).innerHTML = rows
    .map(
      ([label, value]) => `
        <article class="summary-card">
          <span class="metric-label">${escapeHtml(label)}</span>
          <strong>${escapeHtml(String(value ?? "-"))}</strong>
        </article>
      `,
    )
    .join("");
}

function renderList(containerId, items, fallbackText) {
  const html = (items || []).length
    ? `<ul class="bullet-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : emptyState(fallbackText);
  $(containerId).innerHTML = html;
}

function renderTagList(items) {
  if (!items || !items.length) return emptyState("No items.");
  return `
    <div class="pill-row">
      ${items.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("")}
    </div>
  `;
}

function infoCard(title, body, kicker = "") {
  return `
    <article class="teacher-card">
      ${kicker ? `<div class="card-topline"><span class="card-tag">${escapeHtml(kicker)}</span></div>` : ""}
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </article>
  `;
}

function renderTopMetrics() {
  $("metric-profile").textContent = state.profile;
  const library = activeTeacherLibrary();
  $("metric-imported-teachers").textContent = library
    ? `${library.display_name_zh || "A股趋势回调与突破教师库"} · ${library.teacher_count ?? state.overview?.teacher_count ?? "-"} 位老师`
    : String(state.overview?.teacher_count ?? "-");
  $("metric-runs").textContent = String(state.runSpecs?.length ?? 0);
  $("metric-runtime").textContent = runtimeStatusLabel();
}

function isLiveRuntimeHealthy() {
  return Boolean(state.liveConfig?.local_service?.service_healthy);
}

function runtimeStatusLabel() {
  if (!state.liveConfig) return "本地 GPT-OSS：状态检查失败";
  const local = state.liveConfig.local_service || {};
  if (local.service_healthy) return "本地 GPT-OSS：可用于 live scoring";
  if (local.health_error || local.load_error) return "本地 GPT-OSS：状态检查失败";
  return "本地 GPT-OSS：未启动";
}

function renderSimpleNaturalRuntime() {
  const card = $("simple-natural-runtime");
  if (!card) return;
  const liveReady = isLiveRuntimeHealthy();
  const local = state.liveConfig?.local_service || {};
  card.classList.toggle("is-ready", liveReady);
  card.classList.toggle("is-offline", !liveReady);
  card.innerHTML = liveReady
    ? `
      <strong>本地 GPT-OSS 可用</strong>
      <p>可以直接看单只股票，也可以上传候选信号。结果会显示综合分、两个时间窗口和四个老师模型的简短看法。</p>
    `
    : `
      <strong>本地 GPT-OSS 暂未连接</strong>
      <p>${escapeHtml(local.health_error || local.load_error || "当前只能做提示词预览、输入校验、界面演示或历史样本复核，不会真实调用模型。")}</p>
    `;
}

function simpleTaskType() {
  return state.taskState?.task_type || "";
}

function simpleScoringMode() {
  return $("simple-scoring-mode")?.value || "prompt_only";
}

function readableScoringMode(mode) {
  const labels = {
    prompt_only: "prompt_only，仅生成提示词预览",
    dry_run: "dry_run，仅校验输入",
    mock: "mock，仅用于界面演示",
    archived_replay: "历史样本复核",
    live: "live，会调用本地 GPT-OSS",
  };
  return labels[mode] || mode || "-";
}

function isMostlyChinese(text) {
  return /[\u4e00-\u9fff]/.test(String(text || ""));
}

function teacherDisplayNameZh(teacher = {}) {
  if (teacher.display_name_zh) return teacher.display_name_zh;
  const merged = `${teacher.round_id || ""} ${teacher.title || ""}`.toLowerCase();
  if (merged.includes("038") || merged.includes("breakout")) return "突破延续老师";
  if (merged.includes("042")) return "均线回调老师";
  if (merged.includes("050")) return "动量回调老师";
  if (merged.includes("026")) return "量能-KDJ 回调老师";
  if (merged.includes("pullback")) return "趋势回调老师";
  return "综合技术形态老师";
}

function teacherScoreBandZh(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "匹配度未知";
  if (value >= 70) return "高度匹配";
  if (value >= 55) return "中等偏强";
  if (value >= 40) return "部分匹配";
  if (value >= 25) return "匹配偏弱";
  return "明显不匹配";
}

function teacherNoteZh(teacher = {}) {
  if (teacher.note_zh) return teacher.note_zh;
  const name = teacherDisplayNameZh(teacher);
  const band = teacherScoreBandZh(teacher.score);
  if (name.includes("突破")) return `${band}。关注趋势突破后的延续性、量能确认和波动率配合。`;
  if (name.includes("均线")) return `${band}。关注贴近 MA20 后的回调承接、波动率收敛和量价配合。`;
  if (name.includes("动量回调")) return `${band}。关注短线回调后动量能否重新转强。`;
  if (name.includes("量能-KDJ")) return `${band}。关注量能动量和 KDJ 回调结构是否同时落在舒适区。`;
  return `${band}。关注多因子技术形态是否落在该老师的舒适区。`;
}

function renderRuntimeAwareControls() {
  const liveReady = isLiveRuntimeHealthy();
  const liveOption = $("simple-scoring-mode")?.querySelector('option[value="live"]');
  if (liveOption) {
    liveOption.disabled = !liveReady;
    liveOption.textContent = liveReady
      ? "live scoring (local GPT-OSS ready)"
      : "live scoring (local GPT-OSS offline)";
  }
  if ($("simple-scoring-runtime-note")) {
    $("simple-scoring-runtime-note").classList.toggle("warn-card", !liveReady);
    $("simple-scoring-runtime-note").classList.toggle("ok-card", liveReady);
    $("simple-scoring-runtime-note").innerHTML = liveReady
      ? `
        <strong>本地 GPT-OSS 已连接</strong>
        <p>
          当前本地模型服务健康。选择实时评分会真实调用 GPT-OSS；评分仍然是研究辅助，不构成投资建议。
        </p>
      `
      : `
        <strong>本地 GPT-OSS 未连接</strong>
        <p>
          当前未连接到本地 GPT-OSS。你仍可使用提示词预览、输入校验、mock 或历史样本复核；
          live scoring 需要先启动本地 vLLM 服务。
        </p>
      `;
  }
  renderSimpleNaturalRuntime();
  renderSimpleScoringModeControls();
}

function renderSimpleScoringModeControls() {
  const mode = simpleScoringMode();
  const liveReady = isLiveRuntimeHealthy();
  const button = $("simple-scoring-button");
  const badge = $("simple-scoring-live-badge");
  if (!button) return;
  if (mode === "live") {
    if (liveReady) {
      button.disabled = false;
      button.textContent = "调用本地 GPT-OSS 进行评分";
      button.classList.add("live-mode");
      if (badge) {
        badge.classList.remove("is-hidden");
        badge.textContent = "会真实调用模型";
      }
    } else {
      button.disabled = true;
      button.textContent = "本地 GPT-OSS 未连接，无法 live scoring";
      button.classList.remove("live-mode");
      if (badge) {
        badge.classList.remove("is-hidden");
        badge.textContent = "请切换到 prompt_only 或 dry_run";
      }
    }
    return;
  }
  button.disabled = false;
  button.textContent = "生成评分预览 / 校验产物";
  button.classList.remove("live-mode");
  if (badge) badge.classList.add("is-hidden");
}

function renderStatusRibbon() {
  $("status-project").textContent = state.activeRun?.project_id || state.projectView?.project_id || "-";
  $("status-dataset").textContent = state.activeRun?.dataset_id || state.projectView?.dataset_id || "-";
  $("status-run").textContent = state.activeRun?.run_id || state.projectView?.draft_run_id || "-";
  const importedAllowed = Boolean(state.projectView?.allow_imported_fallback);
  const demoAllowed = Boolean(state.projectView?.allow_demo_fallback);
  $("status-fallback").textContent = `系统老师库备用=${importedAllowed ? "on" : "off"} · demo=${demoAllowed ? "on" : "off"}`;
}

function renderSimpleTaskState() {
  const task = state.taskState || {};
  const runStatus = state.simpleRunStatus || {};
  const workflowCard = runStatus.task_card || {};
  const data = task.dataset_summary || {};
  const dateRange = data.date_range
    ? `${data.date_range.start || "-"} -> ${data.date_range.end || "-"}`
    : "-";
  const missingColumns = data.missing_columns || task.missing_columns || [];
  const statusTone = workflowCard.workflow_status === "failed"
    ? "failed"
    : workflowCard.fallback_used || task.fallback_used
      ? "fallback"
      : missingColumns.length || (task.missing_fields || []).length
        ? "warning"
        : "ok";
  const statusZh = {
    completed: "完成",
    failed: "失败",
    running: "运行中",
    pending: "等待中",
    queued: "排队中",
    spec_only: "未启动",
    unavailable: "不可用",
  };
  const nextAction = workflowCard.next_action || task.next_action || task.recommended_next_action_zh || "先告诉我你的研究目标";
  const uploaded = Boolean(task.dataset_manifest_path || data.manifest_path || data.rows);
  const manifestReady = Boolean(task.dataset_manifest_path || data.manifest_path || task.artifact_exists?.dataset_manifest_json);
  const workflowStatus = workflowCard.workflow_status || runStatus.workflow_status || task.workflow_status || "未开始";
  const scoringMode = task.scoring_mode || simpleScoringMode();
  const section = (title, tone, rows) => `
    <article class="task-state-section ${escapeHtml(tone || "")}">
      <div class="task-state-section-head">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(tone || "info")}</span>
      </div>
      <div class="task-state-kv">
        ${rows
          .map(([label, value]) => `
            <div>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(String(value ?? "-"))}</strong>
            </div>
          `)
          .join("")}
      </div>
    </article>
  `;
  const technicalRows = [
    ["run_id", task.run_id || state.activeRun?.run_id || "-"],
    ["artifact full path", workflowCard.latest_artifact || task.latest_artifact || "-"],
    ["completed stages", (workflowCard.completed_stages || []).join(", ") || "-"],
    ["failed stage raw name", workflowCard.failed_stage || task.failed_stage || "-"],
    ["lesson_source raw key", task.lesson_source || "-"],
    ["teacher_source raw key", task.teacher_source || "-"],
    ["model_called raw bool", task.model_called === false ? "false" : task.model_called ? "true" : "-"],
    ["result_valid_for_research raw bool", task.result_valid_for_research === false ? "false" : task.result_valid_for_research ? "true" : "-"],
    ["fallback raw fields", `used=${workflowCard.fallback_used || task.fallback_used ? "true" : "false"}; reason=${workflowCard.fallback_reason || task.fallback_reason || "-"}`],
  ];
  $("simple-task-state-cards").innerHTML = [
    section("当前任务", task.task_type ? "ok" : "warning", [
      ["任务类型", task.task_type_zh || "等待用户输入"],
      ["置信度", task.confidence != null ? Number(task.confidence).toFixed(2) : "-"],
      ["下一步", nextAction],
    ]),
    section("数据状态", missingColumns.length ? "warning" : data.readiness || task.data_readiness ? "ok" : "warning", [
      ["是否已上传", uploaded ? "是" : "否"],
      ["是否通过检查", data.valid || data.readiness === "ready" || task.data_readiness === "ready" ? "是" : "待检查"],
      ["缺少字段", missingColumns.length ? missingColumns.join(", ") : "无 / 待检查"],
      ["数据清单", manifestReady ? "已生成" : "未生成"],
    ]),
    section("运行状态", statusTone, [
      ["状态", statusZh[workflowStatus] || workflowStatus || "未开始"],
      ["当前阶段", workflowCard.current_stage || task.current_stage || "等待开始"],
      ["失败原因", workflowCard.failed_stage || task.failed_stage || "-"],
    ]),
    section("Scoring 状态", task.scoring_status === "validation_failed" ? "failed" : task.scoring_status ? "ok" : "warning", [
      ["评分模式", readableScoringMode(scoringMode)],
      ["是否调用模型", task.model_called ? "是" : "否 / 未评分"],
      ["总分", task.last_score ?? "暂无"],
      ["经验规则来源", task.lesson_source || "待确定"],
    ]),
    section("风险与来源", workflowCard.fallback_used || task.fallback_used ? "fallback" : "ok", [
      ["是否发生回退", workflowCard.fallback_used || task.fallback_used ? "是" : "否 / 未发生"],
      ["回退原因", workflowCard.fallback_reason || task.fallback_reason || "-"],
      ["是否构成投资建议", "否，仅用于研究辅助"],
    ]),
    `
      <details class="details-block task-technical-details">
        <summary>展开技术详情</summary>
        <div class="task-state-kv">
          ${technicalRows
            .map(([label, value]) => `
              <div>
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(String(value ?? "-"))}</strong>
              </div>
            `)
            .join("")}
        </div>
      </details>
    `,
  ].join("");

  const warnings = task.boundary_warnings_zh || [
    "只有当你明确要求查看具体股票时，系统才会尝试联网补齐最近 K 线。",
    "单只股票请求会自动生成 K 线技术因子；批量自定义信号仍建议上传结构化 features。",
    "评分结果是研究辅助，不构成投资建议。",
  ];
  const missing = task.missing_fields || [];
  $("simple-boundary-warnings").innerHTML = `
    <strong>${workflowCard.workflow_status === "failed" ? "Workflow 失败" : missing.length ? "缺失信息" : "系统边界"}</strong>
    ${workflowCard.workflow_status === "completed" && task.task_type !== "scoring_only" ? "<p>本次 workflow 已完成。你可以查看 Teacher Zoo、Final Lesson Set，或上传新信号进行评分。</p>" : ""}
    ${workflowCard.workflow_status === "failed" ? `<p>本次 workflow 在 ${escapeHtml(workflowCard.failed_stage || "未知")} 阶段失败。请查看专业日志或 Audit Trail。</p>` : ""}
    ${missing.length ? `<p>${escapeHtml(missing.join(", "))}</p>` : ""}
    <ul class="bullet-list">${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
  `;
  renderSimpleToolVisibility();
}

function renderSimpleToolVisibility() {
  const taskType = simpleTaskType();
  const toolEmpty = $("simple-tool-empty");
  const tools = document.querySelectorAll("[data-simple-tool]");
  const show = new Set();
  if (taskType === "full_research_pipeline") {
    show.add("upload");
    show.add("kline");
  } else if (taskType === "scoring_only") {
    show.add("upload");
    show.add("scoring");
  } else if (taskType === "online_kline_download") {
    show.add("kline");
  }
  tools.forEach((tool) => {
    const visible = show.has(tool.dataset.simpleTool);
    tool.classList.toggle("is-hidden", !visible);
    tool.classList.toggle("is-secondary-tool", taskType === "full_research_pipeline" && tool.dataset.simpleTool === "kline");
  });
  if (toolEmpty) {
    const helper = {
      full_research_pipeline: "已识别为完整研究流程：请上传 OHLCV/amount 历史行情；如需联网下载，可展开在线 K 线工具。",
      scoring_only: "已识别为信号评分：请上传或粘贴候选信号，至少包含 signal_id、signal_date、symbol、signal_type 和 features。",
      online_kline_download: "已识别为需要补行情：我会在你明确提出股票查看需求时尝试联网取最近 K 线。",
      imported_asset_demo: "已识别为系统已有老师库演示：不强制上传数据，可直接查看老师库、经验规则或做轻量评分。",
      artifact_review: "已识别为已有产物复盘：不强制上传数据，可点击推荐动作进入专业复盘。",
    };
    toolEmpty.classList.toggle("is-hidden", Boolean(show.size));
    toolEmpty.querySelector("p").textContent = helper[taskType] || "选择任务后，我会展开对应的数据上传、在线 K 线下载或信号评分工具。";
  }
  const uploadTitle = $("simple-upload-title");
  const uploadCopy = $("simple-upload-copy");
  if (uploadTitle && uploadCopy) {
    if (taskType === "full_research_pipeline") {
      uploadTitle.textContent = "上传市场行情数据并生成数据清单";
      uploadCopy.textContent = "完整研究流程需要 date、symbol、open、high、low、close、volume、amount 等历史行情字段。";
    } else {
      uploadTitle.textContent = "上传候选信号并生成数据清单";
      uploadCopy.textContent = "候选信号至少需要包含 signal_id、signal_date、symbol、signal_type 和 features。旧字段 date 可以作为 signal_date 的别名自动识别。";
    }
  }
}

function renderSimpleActions() {
  const actions = [...(state.chatRecommendedActions || [])];
  const runStatus = state.simpleRunStatus || {};
  const post = runStatus.post_workflow_actions || {};
  if (runStatus.workflow_status === "completed") {
    (post.completed || []).forEach((item, idx) => {
      actions.push({
        action_id: `post_completed_${idx}`,
        label_zh: item.label_zh,
        type: "navigate",
        enabled: true,
        reason_zh: "workflow 已完成后的建议操作。",
        target_api: "/console",
        expert_link: item.expert_link,
      });
    });
  }
  if (runStatus.workflow_status === "failed") {
    (post.failed || []).forEach((item, idx) => {
      actions.push({
        action_id: `post_failed_${idx}`,
        label_zh: item.label_zh,
        type: "navigate",
        enabled: true,
        reason_zh: "workflow 失败后的恢复操作。",
        target_api: "/console",
        expert_link: item.expert_link,
      });
    });
  }
  $("simple-actions").innerHTML = actions.length
    ? actions
        .map(
          (action) => {
            const label = action.action_id === "score_signal" ? "开始信号评分" : action.label_zh || action.action_id;
            const reason = action.action_id === "score_signal"
              ? `当前模式：${readableScoringMode(simpleScoringMode())}。${simpleScoringMode() === "live" ? "会调用本地 GPT-OSS。" : "不会调用模型。"}`
              : action.reason_zh || "";
            return `
            <button class="simple-action-card ${action.enabled ? "" : "is-disabled"}" data-simple-action="${escapeHtml(action.action_id)}" data-expert-link="${escapeHtml(action.expert_link || "")}" ${action.enabled ? "" : "disabled"}>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(action.type || "-")}</strong>
              <p>${escapeHtml(reason)}</p>
              <details class="action-tech-details">
                <summary>技术信息</summary>
                <small>${escapeHtml(action.target_api || "")}</small>
              </details>
            </button>
          `;
          },
        )
        .join("")
    : emptyState("发送需求后，系统会生成可点击的下一步动作。");

  document.querySelectorAll("[data-simple-action]").forEach((button) => {
    button.addEventListener("click", () => handleSimpleAction(button.dataset.simpleAction, button.dataset.expertLink));
  });
}

function renderSimpleAgentTimeline() {
  const runStatus = state.simpleRunStatus || {};
  const nodes = runStatus.timeline_nodes || [];
  const simpleStages = buildSimpleStageTimeline(nodes);
  $("simple-agent-timeline").innerHTML = simpleStages.map(timelineNodeCard).join("");
  const tech = $("simple-agent-tech-timeline");
  if (tech) {
    tech.innerHTML = nodes.length
      ? nodes.map(timelineNodeCard).join("")
      : emptyState("暂无真实 workflow run status。这里不会伪装 Agent 已执行，等待 workflow 启动后再显示技术阶段。");
  }
  wireTimelineNodeClicks(simpleStages, false);
  wireTimelineNodeClicks(nodes, true);
  const selected = nodes.find((node) => node.node_id === state.simpleSelectedTimelineNodeId) || null;
  renderTimelineDetail(selected);
}

function buildSimpleStageTimeline(nodes = []) {
  const task = state.taskState || {};
  const exists = task.artifact_exists || {};
  const workflowStatus = state.simpleRunStatus?.workflow_status || task.workflow_status || "";
  const hasTask = Boolean(task.task_type);
  const hasManifest = Boolean(exists.dataset_manifest_json || task.dataset_manifest_path || task.dataset_summary?.manifest_path);
  const hasRunSpec = Boolean(exists.run_spec_json || task.run_spec_path);
  const hasWorkflow = Boolean(exists.workflow_result_json || task.workflow_result_path || nodes.length || workflowStatus);
  const hasScoring = Boolean(task.scoring_status || state.simpleScoringResult);
  const workflowStageStatus = (() => {
    if (!hasWorkflow) return hasRunSpec ? "pending" : "not_started";
    if (["completed", "failed", "running", "pending", "queued", "fallback"].includes(workflowStatus)) return workflowStatus;
    return hasWorkflow ? "completed" : "not_started";
  })();
  const scoringStatus = (() => {
    if (task.scoring_status === "validation_failed") return "failed";
    if (hasScoring) return "completed";
    return simpleTaskType() === "scoring_only" ? "pending" : "not_started";
  })();
  return [
    {
      node_id: "simple_task_intake",
      agent_name: "理解任务",
      mapped_stage: hasTask ? "已识别用户意图" : "等待开始",
      status: hasTask ? "completed" : "pending",
      summary_path: "",
      expert_link: "simple",
    },
    {
      node_id: "simple_dataset",
      agent_name: "准备数据",
      mapped_stage: hasManifest ? "数据清单已生成" : "等待上传数据或选择系统已有老师库",
      status: hasManifest ? "completed" : hasTask ? "pending" : "not_started",
      summary_path: task.dataset_manifest_path || "",
      expert_link: "dataset-lab",
    },
    {
      node_id: "simple_run_spec",
      agent_name: "生成研究配置",
      mapped_stage: hasRunSpec ? "run_spec 已生成" : "等待生成 run_spec",
      status: hasRunSpec ? "completed" : hasManifest ? "pending" : "not_started",
      summary_path: task.run_spec_path || "",
      expert_link: "workflow-monitor",
    },
    {
      node_id: "simple_workflow",
      agent_name: "运行研究流程",
      mapped_stage: state.simpleRunStatus?.task_card?.current_stage || task.current_stage || "等待 workflow",
      status: workflowStageStatus,
      summary_path: state.simpleRunStatus?.task_card?.latest_artifact || task.workflow_result_path || "",
      fallback_used: state.simpleRunStatus?.task_card?.fallback_used || task.fallback_used,
      fallback_reason: state.simpleRunStatus?.task_card?.fallback_reason || task.fallback_reason || "",
      expert_link: "workflow-monitor",
    },
    {
      node_id: "simple_scoring",
      agent_name: "生成结果 / 评分",
      mapped_stage: hasScoring ? "评分结果已生成" : "等待评分或查看结果",
      status: scoringStatus,
      summary_path: task.last_scoring_result_path || "",
      expert_link: "scoring",
    },
  ];
}

function timelineNodeCard(node) {
  const status = node.status || "not_started";
  const icon = status === "completed" ? "✓" : status === "failed" ? "!" : status === "fallback" ? "⚠" : status === "running" ? "…" : "•";
  return `
    <button class="timeline-agent-node ${escapeHtml(status)}" data-timeline-node="${escapeHtml(node.node_id)}" type="button">
      <span class="timeline-icon">${escapeHtml(icon)}</span>
      <strong>${escapeHtml(node.agent_name || node.label || "-")}</strong>
      <small>${escapeHtml(status)}</small>
      <p>${escapeHtml(node.mapped_stage || "暂无")}</p>
    </button>
  `;
}

function wireTimelineNodeClicks(nodes, technical = false) {
  const root = technical ? $("simple-agent-tech-timeline") : $("simple-agent-timeline");
  if (!root) return;
  root.querySelectorAll("[data-timeline-node]").forEach((button) => {
    button.addEventListener("click", () => {
      state.simpleSelectedTimelineNodeId = button.dataset.timelineNode;
      const node = nodes.find((item) => item.node_id === state.simpleSelectedTimelineNodeId) || null;
      if (technical) {
        renderTimelineDetail(node);
      } else {
        setWorkspace(node?.expert_link || "full-pipeline");
      }
    });
  });
}

function renderTimelineDetail(node) {
  if (!node) {
    $("simple-agent-detail").innerHTML = emptyState("展开“查看技术阶段”后，可以查看真实 stage-level Agent 映射详情。");
    return;
  }
  const outputArtifacts = node.output_artifacts || [];
  const inputArtifacts = node.input_artifacts || [];
  $("simple-agent-detail").innerHTML = `
    <div class="timeline-detail-head">
      <div>
        <p class="panel-kicker">Node Detail</p>
        <h3>${escapeHtml(node.agent_name || "-")}</h3>
      </div>
      <span class="timeline-status-pill ${escapeHtml(node.status || "not_started")}">${escapeHtml(node.status || "not_started")}</span>
    </div>
    <div class="details-kv-grid">
      <div class="kv"><span>mapped_stage</span><strong>${escapeHtml(node.mapped_stage || "暂无")}</strong></div>
      <div class="kv"><span>fallback_used</span><strong>${escapeHtml(node.fallback_used ? "true" : "false")}</strong></div>
      <div class="kv"><span>fallback_reason</span><strong>${escapeHtml(node.fallback_reason || "暂无")}</strong></div>
      <div class="kv"><span>fallback_source</span><strong>${escapeHtml(node.fallback_source || "暂无")}</strong></div>
      <div class="kv"><span>error_message</span><strong>${escapeHtml(node.error_message || "暂无")}</strong></div>
      <div class="kv"><span>summary_path</span><strong>${escapeHtml(node.summary_path || "暂无")}</strong></div>
      <div class="kv"><span>started_at</span><strong>${escapeHtml(node.started_at || "暂无")}</strong></div>
      <div class="kv"><span>finished_at</span><strong>${escapeHtml(node.finished_at || "暂无")}</strong></div>
    </div>
    <div class="artifact-mini-list">
      <strong>input_artifacts</strong>
      ${inputArtifacts.length ? inputArtifacts.map((path) => `<code>${escapeHtml(path)}</code>`).join("") : "<p>暂无</p>"}
      <strong>output_artifacts</strong>
      ${outputArtifacts.length ? outputArtifacts.map((path) => `<code>${escapeHtml(path)}</code>`).join("") : "<p>暂无</p>"}
    </div>
    <button class="action-button secondary timeline-expert-link" type="button" data-expert-link="${escapeHtml(node.expert_link || "full-pipeline")}">打开对应 Expert 页面</button>
  `;
  const button = $("simple-agent-detail").querySelector("[data-expert-link]");
  if (button) {
    button.addEventListener("click", () => {
      setWorkspace(button.dataset.expertLink || "full-pipeline");
    });
  }
}

function renderSimpleArtifactLinks() {
  const links = [...(state.chatArtifactLinks || [])];
  const scoringPaths = state.simpleScoringResult?.artifact_paths || {};
  Object.entries(scoringPaths).forEach(([label, path]) => {
    if (path) links.push({ label, path, exists: true, source: "scoring_artifact" });
  });
  $("simple-artifact-links").innerHTML = links.length
    ? links
        .map(
          (link) => {
            const meta = readableArtifactMeta(link);
            return `
            <article class="artifact-readable-card ${link.exists ? "exists" : "pending"}">
              <div>
                <span>${escapeHtml(link.exists ? "已生成" : "等待生成")}</span>
                <strong>${escapeHtml(meta.title)}</strong>
                <p>${escapeHtml(meta.description)}</p>
              </div>
              <div class="button-row compact">
                <button type="button" class="ghost-button mini" data-expert-link="${escapeHtml(meta.expertLink)}">查看专业详情</button>
              </div>
              <details class="artifact-path-details">
                <summary>展开技术路径</summary>
                <code>${escapeHtml(link.path || "-")}</code>
              </details>
            </article>
          `;
          },
        )
        .join("")
    : emptyState("发送需求后会显示 session、task_state、run_spec、workflow_result 等同一 run 的产物路径。");
  $("simple-artifact-links").querySelectorAll("[data-expert-link]").forEach((button) => {
    button.addEventListener("click", () => setWorkspace(button.dataset.expertLink || "provenance"));
  });
}

function readableArtifactMeta(link) {
  const label = String(link.label || "");
  const lower = `${label} ${link.path || ""}`.toLowerCase();
  if (lower.includes("dataset_manifest")) {
    return { title: "数据清单已生成", description: "记录数据字段、日期范围、缺失字段和隔离状态。", expertLink: "dataset-lab" };
  }
  if (lower.includes("run_spec") || lower.includes("research_campaign")) {
    return { title: "运行配置已生成", description: "保存英文 canonical run_spec / research_campaign。", expertLink: "workflow-monitor" };
  }
  if (lower.includes("workflow_result")) {
    return { title: "Workflow 结果已生成", description: "记录本次流程执行状态、产物和 fallback。", expertLink: "workflow-monitor" };
  }
  if (lower.includes("live_cache") || lower.includes("raw_response")) {
    return { title: "模型原始输出已保存", description: "live scoring 的 raw_response 保存在技术产物中。", expertLink: "scoring" };
  }
  if (lower.includes("live_saved_run") || lower.includes("parsed_result")) {
    return { title: "解析结果已保存", description: "保存 live scoring 的 parsed result 与评分摘要。", expertLink: "scoring" };
  }
  if (lower.includes("scoring_provenance") || lower.includes("provenance")) {
    return { title: "来源追溯已保存", description: "记录 lesson_source、teacher_source、fallback_used 和 fallback_reason。", expertLink: "provenance" };
  }
  if (lower.includes("scoring")) {
    return { title: "评分结果已保存", description: "保存本次 scoring 输入、校验和结果产物。", expertLink: "scoring" };
  }
  if (lower.includes("chat/")) {
    return { title: "对话状态已保存", description: "保存 session、task_state 或 messages，用于同一 run 复盘。", expertLink: "provenance" };
  }
  return { title: label || "产物", description: safeArtifactName(link.path || "") || "等待生成。", expertLink: "provenance" };
}

function renderSimpleChatThread() {
  const initial = `
    <article class="chat-bubble assistant">
      <span>QuantApprentice</span>
      <p>你好，直接告诉我你想看的股票就可以。比如：“今天帮我看看 000001 这只股票”。我会尽量自动补齐最近 K 线，给出综合分、近 60 日 / 近 120 日视角、四个老师模型的看法和简短理由。结果仅用于研究辅助，不构成投资建议。</p>
    </article>
  `;
  $("simple-chat-messages").innerHTML =
    initial +
    (state.chatMessages || [])
      .map(
        (msg) => `
          <article class="chat-bubble ${msg.role === "user" ? "user" : "assistant"}">
            <span>${msg.role === "user" ? "You" : "QuantApprentice"}</span>
            <p>${escapeHtml(msg.content || "")}</p>
          </article>
        `,
      )
      .join("") +
    (state.simpleTyping
      ? `
        <article class="chat-bubble assistant typing">
          <span>QuantApprentice</span>
          <p><i></i><i></i><i></i> 正在整理任务状态...</p>
        </article>
      `
      : "");
  $("simple-chat-messages").scrollTop = $("simple-chat-messages").scrollHeight;
}

function renderExpertChatThread() {
  const root = $("expert-chat-messages");
  if (!root) return;
  const initial = `
    <article class="chat-bubble assistant">
      <span>QuantApprentice Pro</span>
      <p>这里是专业研究工作台。你可以上传市场数据，也可以直接描述研究目标；我会帮你整理任务、检查数据、准备 run spec，并把 workflow 进度放到右侧 Agent 时间线里。</p>
    </article>
  `;
  root.innerHTML =
    initial +
    (state.expertChatMessages || [])
      .map(
        (msg) => `
          <article class="chat-bubble ${msg.role === "user" ? "user" : "assistant"}">
            <span>${msg.role === "user" ? "You" : "QuantApprentice Pro"}</span>
            <p>${escapeHtml(msg.content || "")}</p>
          </article>
        `,
      )
      .join("") +
    (state.expertTyping
      ? `
        <article class="chat-bubble assistant typing">
          <span>QuantApprentice Pro</span>
          <p><i></i><i></i><i></i> 正在整理研究任务...</p>
        </article>
      `
      : "");
  root.scrollTop = root.scrollHeight;
}

function stageStatusClass(status) {
  const normalized = String(status || "").toLowerCase();
  if (["ready", "completed", "ok", "generated"].includes(normalized)) return "ok-card";
  if (["available", "library_ready"].includes(normalized)) return "accent-card";
  if (["failed", "invalid", "error"].includes(normalized)) return "warn-card";
  if (["running", "pending", "queued", "partial"].includes(normalized)) return "accent-card";
  return "";
}

function expertStatusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  if (["ready", "generated", "ok", "completed"].includes(normalized)) return "已就绪";
  if (["available", "library_ready"].includes(normalized)) return "老师库可用";
  if (["running"].includes(normalized)) return "运行中";
  if (["pending", "queued", "partial"].includes(normalized)) return "进行中";
  if (["failed", "invalid", "error"].includes(normalized)) return "需要处理";
  if (normalized.includes("未") || normalized.includes("no")) return status || "未开始";
  return status || "未开始";
}

function renderExpertStageGrid() {
  const root = $("expert-stage-grid");
  if (!root) return;
  const manifest = state.datasetManifest || {};
  const run = state.runMonitor || {};
  const zoo = state.teacherZoo || {};
  const lesson = state.lessonSet || {};
  const scoring = state.simpleScoringResult || {};
  const datasetStatus = manifest.exists === false ? "未准备" : manifest.data_readiness || (manifest.valid ? "ready" : "未准备");
  const workflowStatus = run.workflow_status || "未启动";
  const selectedCount = zoo.selected_teachers_for_inner_loop?.length ?? 0;
  const frozenCount = zoo.current_workflow_frozen_teachers?.length ?? 0;
  const library = activeTeacherLibrary();
  const libraryCount = library?.teacher_count ?? state.importedTeachers?.length ?? 0;
  const lessonStatus = lesson.final_lesson_state_json ? "generated" : "未生成";
  const scoringStatus = scoring.mode ? "available" : "未评分";
  const fallbackText = zoo.fallback_reason || scoring.scoring_provenance?.fallback_reason || "";
  const missingCount = (manifest.missing_columns || []).length;
  const datasetBody = manifest.exists === false
    ? "还没有可用数据。你可以上传市场数据，或先使用系统已有老师库做单股/信号评分。"
    : missingCount
      ? `数据已读取，但还缺 ${missingCount} 个必要字段，需要先修正。`
      : `数据可用：${manifest.symbol_count ?? "-"} 个标的，${manifest.row_count ?? "-"} 行记录。`;
  const workflowBody = ["running", "pending", "queued"].includes(String(workflowStatus).toLowerCase())
    ? "流程正在运行，右侧 Agent Activity 会显示真实阶段进度。"
    : workflowStatus === "未启动"
      ? "还没有启动完整研究流程。你可以先在聊天框描述研究目标并上传数据。"
      : `最近一次流程状态：${workflowStatus}。`;
  const teacherLessonBody = selectedCount || lesson.final_lesson_state_json
    ? `已选老师 ${selectedCount} 个；最终经验规则集${lesson.final_lesson_state_json ? "已生成" : "尚未生成"}。`
    : libraryCount
      ? `当前可用「${library?.display_name_zh || "系统已有老师库"}」，包含 ${libraryCount} 位老师；本次流程还未产生新老师库。`
      : "还没有可用老师或最终经验规则集。";
  const scoringBody = scoring.mode
    ? `已有评分结果：${scoring.total_score ?? scoring.summary?.mean_score ?? "-"} 分，模式为 ${scoring.mode}。`
    : "还没有评分结果。生成最终经验规则集后，可上传候选信号或在普通模式看单股评分。";
  const rows = [
    {
      title: "数据",
      status: datasetStatus,
      body: datasetBody,
      action: "查看数据",
      target: "full-pipeline",
    },
    {
      title: "工作流",
      status: workflowStatus,
      body: workflowBody,
      action: "看进度",
      target: "workflow-monitor",
    },
    {
      title: "老师与经验",
      status: selectedCount || lesson.final_lesson_state_json ? "ready" : (libraryCount || frozenCount ? "library_ready" : "未生成"),
      body: teacherLessonBody,
      action: "看老师",
      target: "teacher-zoo",
    },
    {
      title: "评分 / 回测",
      status: scoringStatus,
      body: scoringBody,
      action: "看评分",
      target: "scoring",
    },
    {
      title: "来源与回退",
      status: fallbackText ? "partial" : (scoring.mode || run.workflow_status ? "ready" : "未产生"),
      body: fallbackText ? `发生回退：${fallbackText}` : (scoring.mode || run.workflow_status ? "当前没有发现回退。结果来源会保留到追溯记录中。" : "还没有新的训练或评分结果；来源追溯将在结果生成后显示。"),
      action: "看来源",
      target: "audit-trail",
    },
  ];
  root.innerHTML = rows
    .map(
      (row) => `
        <article class="summary-card expert-stage-card ${stageStatusClass(row.status)}">
          <span class="metric-label">${escapeHtml(expertStatusLabel(row.status))}</span>
          <strong>${escapeHtml(row.title)}</strong>
          <p>${escapeHtml(row.body)}</p>
          <button type="button" class="ghost-button mini" data-expert-stage-target="${escapeHtml(row.target)}">${escapeHtml(row.action)}</button>
        </article>
      `,
    )
    .join("");
  root.querySelectorAll("[data-expert-stage-target]").forEach((button) => {
    button.addEventListener("click", () => {
      if ($("expert-advanced-details")) $("expert-advanced-details").open = true;
      setWorkspace(button.dataset.expertStageTarget || "full-pipeline");
    });
  });
}

function renderExpertAgentTimeline() {
  const root = $("expert-agent-timeline");
  if (!root) return;
  const nodes = state.simpleRunStatus?.timeline_nodes || state.runMonitor?.nodes || [];
  if (nodes.length) {
    root.innerHTML = nodes.map(timelineNodeCard).join("");
    root.querySelectorAll("[data-timeline-node]").forEach((button) => {
      button.addEventListener("click", () => {
        if ($("expert-advanced-details")) $("expert-advanced-details").open = true;
        setWorkspace("workflow-monitor");
      });
    });
    return;
  }
  root.innerHTML = buildSimpleStageTimeline([]).map(timelineNodeCard).join("");
}

function renderExpertWorkbench() {
  renderExpertChatThread();
  renderExpertStageGrid();
  renderExpertAgentTimeline();
}

function renderSimpleMode() {
  renderSimpleTaskState();
  renderSimpleActions();
  renderSimpleAgentTimeline();
  renderSimpleScoringResult();
  renderSimpleArtifactLinks();
  renderSimpleChatThread();
  renderSimpleScoringModeControls();
  renderExpertWorkbench();
}

async function handleSimpleAction(actionId, expertLink = "") {
  const action = (state.chatRecommendedActions || []).find((item) => item.action_id === actionId);
  if (!action && actionId.startsWith("post_")) {
    setWorkspace(expertLink || "full-pipeline");
    return;
  }
  if (!action || !action.enabled) {
    state.chatMessages.push({
      role: "assistant",
      content: "这个动作现在还不能执行。请先用自然语言告诉我你的任务目标，或补齐右侧任务状态里提示的缺失信息。",
    });
    renderSimpleChatThread();
    return;
  }
  try {
    state.simpleTyping = true;
    renderSimpleChatThread();
    if (actionId === "upload_dataset") {
      const file = $("simple-chat-upload-file")?.files?.[0] || $("simple-upload-file").files?.[0];
      if (!file) {
        state.simpleTyping = false;
        state.chatMessages.push({ role: "assistant", content: "请先在左侧选择要上传的 CSV / JSON / Parquet 文件。" });
        renderSimpleChatThread();
        return;
      }
      await executeChatAction("upload_dataset", { file_payload: await readFilePayload(file) });
      if ($("simple-chat-upload-name")) $("simple-chat-upload-name").textContent = `已上传并完成检查：${file.name}`;
      if ($("simple-upload-status")) $("simple-upload-status").textContent = `已上传并完成检查：${file.name}`;
      if ($("simple-chat-upload-submit")) $("simple-chat-upload-submit").classList.add("is-hidden");
      return;
    }
    if (actionId === "start_online_kline_download") {
      const confirmed = window.confirm("这将联网下载 A 股 K 线数据，依赖服务器网络、akshare、tushare 和 TUSHARE_TOKEN。是否继续？");
      if (!confirmed) {
        state.simpleTyping = false;
        state.chatMessages.push({ role: "assistant", content: "已取消在线 K 线下载。系统没有联网。" });
        renderSimpleChatThread();
        return;
      }
      await executeChatAction("start_online_kline_download", {
        confirm: true,
        kline_params: {
          stock_codes: $("simple-kline-codes").value.trim(),
          earliest_date: $("simple-kline-earliest").value.trim() || defaultSimpleKlineEarliestDate(),
          adjust_type: $("simple-kline-adjust").value,
          full_refresh: false,
          update_indexes: $("simple-kline-indexes").checked,
        },
      });
      return;
    }
    if (actionId === "score_stock_code_live") {
      const pendingCodes = state.taskState?.pending_stock_codes || [];
      const stockCodes = pendingCodes.length
        ? pendingCodes.join(", ")
        : ($("simple-kline-codes")?.value || "").trim();
      if (!stockCodes) {
        state.simpleTyping = false;
        state.chatMessages.push({ role: "assistant", content: "我还没有识别到股票代码。你可以直接说：今天帮我看看 000001。" });
        renderSimpleChatThread();
        return;
      }
      await executeChatAction("score_stock_code_live", {
        confirm: true,
        kline_params: {
          stock_codes: stockCodes,
          earliest_date: state.taskState?.pending_kline_earliest_date || $("simple-kline-earliest")?.value?.trim() || defaultSimpleKlineEarliestDate(),
          adjust_type: state.taskState?.pending_adjust_type || $("simple-kline-adjust")?.value || "qfq",
          full_refresh: false,
          update_indexes: false,
        },
      });
      return;
    }
    if (actionId === "create_imported_asset_manifest" || actionId === "use_imported_demo_assets") {
      await executeChatAction("create_imported_asset_manifest", { confirm: true });
      return;
    }
    if (actionId === "generate_run_spec") {
      await executeChatAction("generate_run_spec", { confirm: true });
      return;
    }
    if (actionId === "launch_workflow") {
      if (!window.confirm("确认启动 workflow？这会真正调用后端 pipeline。")) {
        state.simpleTyping = false;
        renderSimpleChatThread();
        return;
      }
      await executeChatAction("launch_workflow", { confirm: true });
      return;
    }
    if (actionId === "score_signal") {
      await runSimpleScoringFlow();
      return;
    }
    if (actionId === "open_expert_monitor") {
      setWorkspace("full-pipeline");
      return;
    }
    if (action.type === "navigate" && action.expert_link) {
      setWorkspace(action.expert_link);
      return;
    }
    if (action.type === "local_ui" || actionId === "continue_upload_signal" || actionId === "batch_score_signal") {
      setWorkspace("simple");
      state.chatMessages.push({
        role: "assistant",
        content: "可以继续在 Simple Scoring 区域粘贴新的 signal JSON，或上传 JSON array / CSV / Parquet 批量文件。",
      });
      renderSimpleChatThread();
      return;
    }
    if (action.type === "disabled_info") {
      state.simpleTyping = false;
      state.chatMessages.push({ role: "assistant", content: action.reason_zh || "这个动作当前仅作为说明，暂不可执行。" });
      renderSimpleChatThread();
      return;
    }
  } catch (error) {
    state.chatMessages.push({ role: "assistant", content: `动作执行失败：${error.message}` });
    renderSimpleChatThread();
  } finally {
    state.simpleTyping = false;
    renderSimpleChatThread();
  }
}

function renderDashboard() {
  const library = activeTeacherLibrary();
  summaryCards("dashboard-quick-status", [
    ["Current profile", state.profile || "-"],
    ["Teacher library", library?.display_name_zh || "系统已有老师库"],
    ["Teachers", library?.teacher_count ?? state.importedTeachers?.length ?? 0],
    ["Lesson versions", state.lessons?.length ?? 0],
    ["Reference markets", state.markets?.length ?? 0],
    ["Local model", runtimeStatusLabel()],
    ["Active run", state.activeRun?.run_id || state.projectView?.draft_run_id || "-"],
  ]);

  $("dashboard-imported-teachers-preview").innerHTML = (state.importedTeachers || []).length
    ? state.importedTeachers.slice(0, 4).map(teacherCard).join("")
    : emptyState("当前 profile 暂未发现可用老师库。");

  $("dashboard-lesson-preview").innerHTML = (state.lessons || []).length
    ? state.lessons
        .slice(0, 4)
        .map((row) => infoCard(row.alias || "lesson", `seed=${row.seed_label || "-"} · final_lesson_state_json ready`, "经验规则版本"))
        .join("")
    : emptyState("当前老师库暂未发现经验规则版本。");

  $("dashboard-market-preview").innerHTML = (state.markets || []).length
    ? state.markets
        .slice(0, 4)
        .map((row) => infoCard(row.alias || "market run", row.window || "window not available", "reference market"))
        .join("")
    : emptyState("当前暂无可复核的历史市场样本。");
}

function renderImportedLessonBrowser() {
  const payload = state.importedLessonDetail || {};
  const summary = payload.summary || [];
  const scopes = payload.scopes || {};
  summaryCards("library-lesson-summary-cards", [
    ["Alias", payload.alias || "-"],
    ["Teacher scopes", Object.keys(scopes).length],
    ["Seed", state.lessons.find((item) => item.alias === payload.alias)?.seed_label || "-"],
    ["Source", "系统已有老师库"],
  ]);
  $("library-lesson-scope-grid").innerHTML = summary.length
    ? summary
        .map(
          (row) => `
            <article class="lesson-card">
              <div class="card-topline">
                <span class="card-tag">${escapeHtml(row.round_id || "-")}</span>
                <span class="pill">系统已有经验规则</span>
              </div>
              <h3>${escapeHtml(row.lesson_name || "Untitled Lesson")}</h3>
              <p>source_round_id=${escapeHtml(row.source_round_id || "-")}</p>
              <div class="pill-row">
                <span class="pill">${escapeHtml(String(row.item_count ?? 0))} items</span>
                <span class="pill">${escapeHtml(String(row.meta_rule_count ?? 0))} meta rules</span>
              </div>
            </article>
          `,
        )
        .join("")
    : emptyState("请选择一个老师库经验版本，查看每位老师对应的经验规则。");
}

function renderScoringControls() {
  const inputMode = $("scoring-input-mode")?.value || "recorded_replay";
  const sourceMode = $("scoring-source")?.value || "imported";
  const recordedFields = $("scoring-recorded-fields");
  const jsonLabel = $("scoring-signal-json")?.closest("label");
  const fileLabel = $("scoring-batch-file")?.closest("label");
  if (recordedFields) recordedFields.hidden = inputMode !== "recorded_replay";
  if (jsonLabel) jsonLabel.hidden = inputMode === "recorded_replay";
  if (fileLabel) fileLabel.hidden = inputMode === "recorded_replay";
  if ($("scoring-lesson-alias")) $("scoring-lesson-alias").disabled = sourceMode !== "imported";
}

function applyContextToForms(payload) {
  const projectId = payload.project_id || payload.projectId || "default-project";
  const datasetId = payload.dataset_id || payload.datasetId || "default-dataset";
  const runId = payload.run_id || payload.draft_run_id || payload.runId || "guided-run";
  const profileId = payload.profile_id || payload.profile || state.profile || "gpt_oss_20b_final";
  const allowImported = Boolean(payload.allow_imported_fallback ?? true);
  const allowDemo = Boolean(payload.allow_demo_fallback ?? false);

  state.profile = profileId;
  for (const [id, value] of [
    ["entry-project-id", projectId],
    ["entry-dataset-id", datasetId],
    ["entry-run-id", runId],
    ["entry-profile", profileId],
    ["project-id", projectId],
    ["dataset-id", datasetId],
    ["run-id", runId],
    ["project-profile", profileId],
  ]) {
    if ($(id)) $(id).value = value;
  }
  if ($("entry-allow-imported-fallback")) $("entry-allow-imported-fallback").checked = allowImported;
  if ($("entry-allow-demo-fallback")) $("entry-allow-demo-fallback").checked = allowDemo;
  if ($("allow-imported-fallback")) $("allow-imported-fallback").checked = allowImported;
  if ($("allow-demo-fallback")) $("allow-demo-fallback").checked = allowDemo;
}

function captureEntryContext() {
  return {
    profile: $("entry-profile").value.trim() || state.profile,
    project_id: $("entry-project-id").value.trim() || "default-project",
    dataset_id: $("entry-dataset-id").value.trim() || "default-dataset",
    run_id: $("entry-run-id").value.trim() || "guided-run",
    allow_imported_fallback: $("entry-allow-imported-fallback").checked,
    allow_demo_fallback: $("entry-allow-demo-fallback").checked,
  };
}

function captureProjectForm() {
  return {
    profile: $("project-profile").value.trim() || state.profile,
    project_id: $("project-id").value.trim() || "default-project",
    dataset_id: $("dataset-id").value.trim() || "default-dataset",
    run_id: $("run-id").value.trim() || "guided-run",
    allow_imported_fallback: $("allow-imported-fallback").checked,
    allow_demo_fallback: $("allow-demo-fallback").checked,
  };
}

function selectedDatasetSource() {
  return $("dataset-source-type")?.value || "upload_local";
}

function renderDatasetSourcePanels() {
  const source = selectedDatasetSource();
  const uploadPanel = $("dataset-source-upload-panel");
  const importedPanel = $("dataset-source-imported-panel");
  const klinePanel = $("dataset-source-kline-panel");
  if (uploadPanel) uploadPanel.hidden = source !== "upload_local";
  if (importedPanel) importedPanel.hidden = source !== "imported_assets";
  if (klinePanel) klinePanel.hidden = source !== "online_kline";
}

function teacherCard(item) {
  const metrics = [
    item.mean_alpha != null ? `mean_alpha=${Number(item.mean_alpha).toFixed(4)}` : "",
    item.nav_cagr != null ? `NAV CAGR=${Number(item.nav_cagr).toFixed(3)}` : "",
    item.positive_years != null ? `positive_years=${item.positive_years}/${item.total_years ?? "?"}` : "",
    item.uplift_mean != null ? `uplift=${Number(item.uplift_mean).toFixed(4)}` : "",
    item.nav_max_drawdown != null ? `MDD=${Number(item.nav_max_drawdown).toFixed(3)}` : "",
  ].filter(Boolean);
  const displayName = teacherDisplayNameZh(item);
  const stateLabel = item.teacher_state || "-";
  const family = item.research_family || item.family || item.sample_template || "";
  const compactMetric = metrics.slice(0, 2).join(" · ");
  const detailRows = [
    ["technical_id", item.round_id || item.frozen_round_id || item.source_round_id || "-"],
    ["teacher_state", stateLabel],
    ["research_family", family || "-"],
    ["sample_template", item.sample_template || "-"],
    ["target_kind", item.target_kind || "-"],
    ["source_round_id", item.source_round_id || "-"],
    ["report_dir", item.report_dir || "-"],
    ["selected_spec_json", item.selected_spec_json || "-"],
  ];
  const reason = item.selection_reason || item.fallback_reason || "";
  return `
    <article class="teacher-card compact-teacher-card">
      <div class="card-topline">
        <span class="card-tag">${escapeHtml(stateLabel)}</span>
        ${sourcePill(item.source_type)}
      </div>
      <h3>${escapeHtml(displayName)}</h3>
      <p>${escapeHtml(family || "A 股技术形态评分老师")}</p>
      ${compactMetric ? `<p class="card-footnote">${escapeHtml(compactMetric)}</p>` : ""}
      <details class="details-block compact teacher-detail-toggle">
        <summary>展开老师详情</summary>
        <div class="details-kv-grid">
          ${detailRows.map(([label, value]) => `<div class="kv"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
        </div>
        ${metrics.length > 2 ? `<p class="card-footnote">${escapeHtml(metrics.slice(2).join(" · "))}</p>` : ""}
        ${reason ? `<p class="card-footnote">${escapeHtml(reason)}</p>` : ""}
      </details>
    </article>
  `;
}

function pathCard(label, value, extra = "") {
  return `
    <div class="path-card">
      <span>${escapeHtml(label)}${extra ? ` · ${escapeHtml(extra)}` : ""}</span>
      <strong>${escapeHtml(value || "-")}</strong>
    </div>
  `;
}

function rawDetails(title, payload) {
  return `
    <details class="details-block">
      <summary>${escapeHtml(title)}</summary>
      <pre class="code-block compact">${escapeHtml(pretty(payload || {}))}</pre>
    </details>
  `;
}

function safeArtifactName(path) {
  if (!path) return "-";
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  return parts.slice(-2).join("/") || String(path);
}

function artifactCard(label, path, payload = {}) {
  const suffix = path && String(path).endsWith(".json")
    ? `<button class="ghost-button open-artifact-json" type="button" data-artifact-path="${escapeHtml(path)}">Open JSON</button>`
    : "";
  return `
    <article class="teacher-score-card">
      <div class="card-topline">
        <span class="card-tag">${escapeHtml(label)}</span>
        ${sourcePill(payload.source_type || "")}
      </div>
      <h3>${escapeHtml(safeArtifactName(path))}</h3>
      <p>${escapeHtml(path || "not generated")}</p>
      ${payload.fallback_reason ? `<p class="card-footnote">fallback_reason=${escapeHtml(payload.fallback_reason)}</p>` : ""}
      ${suffix}
    </article>
  `;
}

function renderCapabilityRequirements(payload) {
  if (!payload) {
    return emptyState("Analyze a task first to see upload requirements.");
  }
  return `
    <div class="capability-grid">
      <article class="capability-card">
        <h3>${escapeHtml(payload.dataset_kind || "-")}</h3>
        <p>${escapeHtml(payload.guidance || "")}</p>
      </article>
      <article class="capability-card">
        <h3>Required columns</h3>
        ${renderTagList(payload.required_columns || [])}
      </article>
      <article class="capability-card">
        <h3>Accepted formats</h3>
        ${renderTagList(payload.accepted_formats || [])}
      </article>
      <article class="capability-card">
        <h3>Optional columns</h3>
        ${renderTagList(payload.optional_columns || [])}
      </article>
    </div>
  `;
}

function renderTaskIntake() {
  const payload = state.taskIntake || {};
  summaryCards("intake-summary-cards", [
    ["Task Type", payload.task_type || "-"],
    ["Confidence", payload.confidence != null ? payload.confidence.toFixed(2) : "-"],
    ["Ready For Run Spec", payload.ready_for_run_spec ? "yes" : "no"],
    ["Next Step", payload.recommended_next_step || "-"],
  ]);
  $("intake-summary-text").textContent = payload.summary || "No intake result yet.";
  $("intake-next-step").innerHTML = payload.recommended_next_step
    ? `<span class="pill accent">${escapeHtml(payload.recommended_next_step)}</span>`
    : "";
  renderList("intake-limitations", payload.limitations || [], "No system limits recorded yet.");
  renderList("intake-questions", payload.clarifying_questions || [], "No clarifying questions. The task description is already fairly complete.");
  $("intake-requirements").innerHTML = renderCapabilityRequirements(payload.dataset_requirements);
  $("dataset-task-type").textContent = payload.task_type
    ? `${payload.task_type} · ${payload.dataset_requirements?.guidance || ""}`
    : "Analyze a task first to see required schema.";
  $("dataset-requirement-cards").innerHTML = renderCapabilityRequirements(payload.dataset_requirements);
}

function renderDatasetManifest() {
  const payload = state.datasetManifest || {};
  const dateRange = payload.date_range
    ? `${payload.date_range.start || "-"} -> ${payload.date_range.end || "-"}`
    : "-";
  summaryCards("dataset-summary-cards", [
    ["Source", payload.source_type || "-"],
    ["Valid", payload.valid ? "yes" : payload.dataset_format ? "no" : "-"],
    ["Readiness", payload.data_readiness || "-"],
    ["Full Pipeline", payload.full_pipeline_ready ? "ready" : "not ready"],
    ["Format", payload.dataset_format || "-"],
    ["Rows", payload.row_count ?? "-"],
    ["Symbols", payload.symbol_count ?? "-"],
    ["Date Range", dateRange],
    ["Isolation", payload.data_isolation_status?.isolated_from_imported_assets ? "isolated" : "-"],
  ]);
  $("dataset-paths").innerHTML = payload.generated_paths
    ? [
        ["Stored Dataset", payload.stored_dataset_path],
        ["Stock Kline Root", payload.stock_kline_root],
        ["Index Kline Root", payload.index_kline_root],
        ["dataset_manifest.json", payload.generated_paths.dataset_manifest_json],
        ["run_spec.json", payload.generated_paths.run_spec_json],
      ]
        .map(
          ([label, value]) => `
            <div class="path-card">
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(value || "-")}</strong>
            </div>
          `,
        )
        .join("")
    : emptyState("还没有 dataset_manifest。请先上传本地数据、选择系统已有老师库，或显式启动在线 K-line 下载。");

  const warnings = payload.warning_reasons || [];
  const failures = payload.fail_reasons || [];
  $("dataset-issues").innerHTML =
    (warnings.length || failures.length)
      ? `
        <div class="issue-stack">
          ${warnings.length ? `<div class="notice-card compact warn-card"><strong>Warnings</strong><ul class="bullet-list">${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
          ${failures.length ? `<div class="notice-card compact warn-card"><strong>Fail Reasons</strong><ul class="bullet-list">${failures.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
        </div>
      `
      : emptyState("暂无 warning / fail reason。数据校验通过时这里会保持为空。");

  $("dataset-columns").innerHTML = payload.columns
    ? `
      <div class="issue-stack">
        <div class="notice-card compact">
          <strong>Required Columns</strong>
          ${renderTagList(payload.required_columns || [])}
        </div>
        <div class="notice-card compact">
          <strong>Missing Columns</strong>
          ${(payload.missing_columns || []).length ? renderTagList(payload.missing_columns || []) : "<p class=\"result-text\">None.</p>"}
        </div>
      </div>
    `
    : emptyState("还没有解析到 columns。生成 dataset_manifest 后会显示 required / missing columns。");

  const previewRows = payload.preview_rows || [];
  if (!previewRows.length) {
    $("dataset-preview-head").innerHTML = "";
    $("dataset-preview-body").innerHTML = `<tr><td colspan="6">${emptyState("还没有 preview rows。请先完成 Dataset Onboarding。")}</td></tr>`;
    return;
  }
  const columns = Object.keys(previewRows[0]);
  $("dataset-preview-head").innerHTML = `
    <tr>${columns.map((key) => `<th>${escapeHtml(key)}</th>`).join("")}</tr>
  `;
  $("dataset-preview-body").innerHTML = previewRows
    .map(
      (row) => `
        <tr>${columns.map((key) => `<td>${escapeHtml(typeof row[key] === "object" ? JSON.stringify(row[key]) : String(row[key] ?? ""))}</td>`).join("")}</tr>
      `,
    )
    .join("");
}

function renderKlineJob() {
  const payload = state.klineJob || {};
  summaryCards("kline-job-summary-cards", [
    ["Status", payload.status || "-"],
    ["Progress", payload.progress != null ? `${Math.round((payload.progress || 0) * 100)}%` : "-"],
    ["Success", payload.success_count ?? "-"],
    ["Failed", payload.failed_count ?? "-"],
  ]);
  const failedCodes = payload.failed_codes || [];
  $("kline-job-issues").innerHTML = payload.job_id
    ? `
      <div class="issue-stack">
        <div class="notice-card compact">
          <strong>Job ID</strong>
          <p class="result-text">${escapeHtml(payload.job_id || "-")}</p>
        </div>
        <div class="notice-card compact ${failedCodes.length ? "warn-card" : ""}">
          <strong>Failed Codes</strong>
          ${failedCodes.length ? renderTagList(failedCodes) : "<p class=\"result-text\">None.</p>"}
        </div>
      </div>
    `
    : emptyState("还没有在线 K-line 下载任务。只有用户显式选择在线下载时，系统才会联网。");
  $("kline-job-logs").textContent = (payload.logs || []).length ? (payload.logs || []).join("\n") : "还没有在线 K-line 下载任务。";
}

function renderDatasetLab() {
  const manifest = state.datasetManifest || {};
  const project = state.projectView || {};
  const dateRange = manifest.date_range
    ? `${manifest.date_range.start || "-"} -> ${manifest.date_range.end || "-"}`
    : "-";
  summaryCards("dataset-lab-summary", [
    ["project_id", project.project_id || manifest.project_id || "-"],
    ["dataset_id", project.dataset_id || manifest.dataset_id || "-"],
    ["run_id", state.activeRun?.run_id || project.draft_run_id || "-"],
    ["profile", state.profile || "-"],
    ["source_type", manifest.source_type || "-"],
    ["data_readiness", manifest.data_readiness || "-"],
    ["row_count", manifest.row_count ?? "-"],
    ["symbol_count", manifest.symbol_count ?? "-"],
    ["date_range", dateRange],
    ["isolation", (manifest.data_isolation_status || project.data_isolation)?.isolated_from_imported_assets ? "isolated" : "-"],
  ]);
  $("dataset-lab-paths").innerHTML = [
    ["dataset_manifest.json", manifest.dataset_manifest_json || manifest.generated_paths?.dataset_manifest_json || project.dataset_manifest_json],
    ["dataset_root", project.dataset_root],
    ["data root", project.dataset_raw_root || manifest.stock_kline_root || manifest.stored_dataset_path],
    ["stock_kline_root", manifest.stock_kline_root || project.dataset_stock_klines_root],
    ["index_kline_root", manifest.index_kline_root || project.dataset_index_klines_root],
    ["cache root", project.dataset_cache_root],
    ["upload root", project.dataset_upload_root],
  ].map(([label, value]) => pathCard(label, value)).join("");
  $("dataset-lab-columns").innerHTML = `
    <div class="issue-stack">
      <div class="notice-card compact"><strong>Required columns</strong>${renderTagList(manifest.required_columns || [])}</div>
      <div class="notice-card compact"><strong>Missing columns</strong>${(manifest.missing_columns || []).length ? renderTagList(manifest.missing_columns || []) : "<p class=\"result-text\">None / not generated.</p>"}</div>
    </div>
  `;
  const warnings = manifest.warning_reasons || manifest.warnings || [];
  const failures = manifest.fail_reasons || [];
  $("dataset-lab-issues").innerHTML = warnings.length || failures.length
    ? `
      <div class="issue-stack">
        ${warnings.length ? `<div class="notice-card compact warn-card"><strong>Warnings</strong><ul class="bullet-list">${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
        ${failures.length ? `<div class="notice-card compact warn-card"><strong>Fail reasons</strong><ul class="bullet-list">${failures.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
      </div>
    `
    : emptyState("暂无 warning / fail reason。");
  const kline = state.klineJob || {};
  summaryCards("dataset-lab-kline-summary", [
    ["job_id", kline.job_id || "-"],
    ["status", kline.status || "-"],
    ["progress", kline.progress != null ? `${Math.round((kline.progress || 0) * 100)}%` : "-"],
    ["success_count", kline.success_count ?? "-"],
    ["failed_count", kline.failed_count ?? "-"],
  ]);
  $("dataset-lab-kline-logs").textContent = (kline.logs || []).join("\n") || "暂无 K-line job。只有用户显式选择在线下载时，系统才会联网。";
  $("dataset-lab-imported").innerHTML = ["imported_assets", "imported_paper_assets"].includes(manifest.source_type)
    ? infoCard("系统已有老师库数据清单", manifest.imported_asset_root || project.imported_asset_root || "-", "系统已有老师库")
    : emptyState("当前 dataset_manifest 不是系统已有老师库清单。若只是想体验已有老师库，请在 Simple Mode 选择系统已有老师库。");
  $("dataset-lab-raw").textContent = manifest.exists === false ? "dataset_manifest.json not generated. 下一步：上传数据或使用系统已有老师库。" : pretty(manifest || {});
}

function renderWorkflowLab() {
  const payload = state.runMonitor || {};
  const chip = $("workflow-lab-status-chip");
  chip.textContent = payload.workflow_status || "No Run Loaded";
  chip.classList.toggle("warn", !["completed", "partial"].includes(payload.workflow_status));
  summaryCards("workflow-lab-strip", [
    ["workflow_status", payload.workflow_status || "-"],
    ["executed_steps", payload.executed_steps ?? "-"],
    ["manual_steps", payload.manual_steps ?? "-"],
    ["failed_steps", payload.failed_steps ?? "-"],
    ["workflow_result.json", payload.workflow_result_json || "-"],
  ]);
  const lookup = {};
  (payload.nodes || []).forEach((node) => {
    lookup[node.node_id] = node;
    lookup[node.label] = node;
  });
  const selectionFallback = state.teacherZoo?.fallback_reason || state.provenance?.teacher_selection_summary?.fallback_reason || "";
  const provenance = state.provenance || {};
  const chain = [
    ["research_spec", "research_spec", "研究目标与 run_spec", provenance.run_spec_json, provenance.research_campaign_json],
    ["outer_loop", "outer_loop", "hypothesis / factor / candidate teacher", provenance.research_campaign_json, ""],
    ["frozen_eval", "teacher_frozen_eval", "frozen evaluation", "", ""],
    ["selection", "TeacherSelectionAgent", "formal teacher selection", "", provenance.teacher_selection_summary_json],
    ["inner_loop", "inner_loop_suite", "warmup / final lesson", provenance.teacher_selection_summary_json, state.lessonSet?.suite_summary_json],
    ["final_lesson_set", "final_lesson_set", "final lesson artifact", state.lessonSet?.suite_summary_json, state.lessonSet?.final_lesson_state_json],
    ["scoring", "SignalScoringAgent", "signal scoring artifacts", state.lessonSet?.final_lesson_state_json, latestScoringArtifacts().byKind.scoring_provenance],
  ];
  $("workflow-lab-chain").innerHTML = chain
    .map(([label, nodeId, description, inputArtifact, defaultOutput]) => {
      const node = lookup[nodeId] || {};
      const nodePayload = node.payload || {};
      const artifact = nodePayload.artifact_json || nodePayload.suite_summary_json || nodePayload.selected_spec_json || nodePayload.final_lesson_artifact_json || defaultOutput || "";
      const fallbackReason = nodeId === "TeacherSelectionAgent" ? selectionFallback : "";
      const runtime = nodePayload.runtime_seconds || nodePayload.elapsed_seconds || node.runtime_seconds || "";
      const summaryPath = nodePayload.summary_json || nodePayload.suite_summary_json || nodePayload.selected_spec_json || "";
      return `
        <article class="timeline-card ${node.status === "failed" ? "warn-card" : ""} ${fallbackReason ? "fallback-card" : ""}">
          <div class="card-topline">
            <span class="card-tag">${escapeHtml(label)}</span>
            <span class="card-tag">${escapeHtml(node.status || "not generated")}</span>
          </div>
          <h3>${escapeHtml(description)}</h3>
          <p><strong>input</strong>: ${escapeHtml(inputArtifact || "not generated")}</p>
          <p><strong>output</strong>: ${escapeHtml(artifact || "not generated")}</p>
          ${summaryPath ? `<p><strong>summary</strong>: ${escapeHtml(summaryPath)}</p>` : ""}
          ${runtime ? `<p><strong>runtime</strong>: ${escapeHtml(String(runtime))}s</p>` : ""}
          ${fallbackReason ? `<p class="card-footnote">fallback_reason=${escapeHtml(fallbackReason)}</p>` : ""}
          ${node.error || nodePayload.error ? `<p class="card-footnote">error=${escapeHtml(node.error || nodePayload.error)}</p>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderTeacherZooLab() {
  const payload = state.teacherZoo || {};
  const importedTeachers = (payload.imported_teachers || []).length ? payload.imported_teachers || [] : state.importedTeachers || [];
  const selected = payload.selected_teachers_for_inner_loop || [];
  const candidate = payload.current_workflow_candidate_teachers || [];
  const validated = payload.current_workflow_validated_teachers || [];
  const frozen = payload.current_workflow_frozen_teachers || [];
  const rejected = payload.current_workflow_rejected_teachers || [];
  const fallbackReason = payload.fallback_reason || "none";
  summaryCards("teacher-lab-summary", [
    ["selection_resolution_source", payload.selection_resolution_source || "-"],
    ["fallback_reason", fallbackReason],
    ["system_teacher_library_teachers", importedTeachers.length],
    ["candidate_teachers", candidate.length],
    ["validated_teachers", validated.length],
    ["frozen_teachers", frozen.length],
    ["selected_for_inner_loop", selected.length],
  ]);
  $("teacher-lab-imported").innerHTML = importedTeachers.length
    ? importedTeachers.map(teacherCard).join("")
    : emptyState("未发现系统已有冻结老师。");
  $("teacher-lab-selected").innerHTML = selected.length
    ? selected.map(teacherCard).join("")
    : emptyState("当前 workflow 尚未生成 selected_teacher_for_inner_loop。请先完成 frozen_eval 与 formal selection；若只是评分，可使用系统已有老师库。");
  $("teacher-lab-candidate").innerHTML = candidate.length
    ? candidate.map(teacherCard).join("")
    : emptyState("当前 workflow 尚未生成 candidate_teacher。请先生成 run_spec 并启动 full pipeline / outer loop。");
  $("teacher-lab-validated").innerHTML = validated.length
    ? validated.map(teacherCard).join("")
    : emptyState("当前 workflow 尚未生成 validated_teacher。outer loop 产出候选后，验证阶段会填充这里。");
  $("teacher-lab-frozen").innerHTML = frozen.length
    ? frozen.map(teacherCard).join("")
    : emptyState("当前 workflow 尚未生成 frozen_teacher / frozen_eval artifact。请运行 frozen eval stage，或查看是否使用了系统已有老师库。");
  $("teacher-lab-rejected").innerHTML = rejected.length
    ? rejected.map(teacherCard).join("")
    : emptyState("当前视图没有 rejected teacher artifact；如果 selection 丢弃了候选，会在后续正式 artifact 中显示。");
}

function lessonScopeCard(row, sourceType = "") {
  return `
    <article class="lesson-card">
      <div class="card-topline">
        <span class="card-tag">${escapeHtml(row.round_id || "-")}</span>
        ${sourcePill(row.source_type || sourceType)}
      </div>
      <h3>${escapeHtml(row.lesson_name || "Untitled Lesson")}</h3>
      <p>source_round_id=${escapeHtml(row.source_round_id || "-")}</p>
      <div class="pill-row">
        <span class="pill">${escapeHtml(String(row.item_count ?? 0))} items</span>
        <span class="pill">${escapeHtml(String(row.meta_rule_count ?? 0))} meta rules</span>
      </div>
    </article>
  `;
}

function renderLessonLab() {
  const current = state.lessonSet || {};
  const imported = state.importedLessonDetail || {};
  const importedScopes = imported.summary || [];
  const currentScopes = current.teacher_scopes || [];
  summaryCards("lesson-lab-summary", [
    ["system_library_alias", imported.alias || "-"],
    ["system_library_teacher_scopes", importedScopes.length],
    ["current_final_lesson_source", current.final_lesson_source || "-"],
    ["current_teacher_scopes", currentScopes.length],
    ["current_final_lesson_state_json", current.final_lesson_state_json || "-"],
    ["suite_summary_json", current.suite_summary_json || "-"],
  ]);
  $("lesson-lab-missing-advice").innerHTML = current.final_lesson_state_json
    ? `<div class="notice-card compact"><strong>Current workflow lesson ready</strong><p>当前 run 已产生 final_lesson_set，可在 Scoring Lab 选择 current_workflow_asset。</p></div>`
    : `<div class="notice-card compact warn-card"><strong>Current workflow lesson not generated</strong><p>如果你要用本次 workflow 的 lesson，请先完成 inner_loop / final_lesson_set；否则可使用系统已有老师库。</p></div>`;
  summaryCards("lesson-lab-imported-summary", [
    ["alias", imported.alias || "-"],
    ["source", imported.alias ? "系统已有老师库" : "-"],
    ["teacher_scopes", importedScopes.length],
    ["selected_seed", state.lessons.find((item) => item.alias === imported.alias)?.seed_label || "-"],
  ]);
  $("lesson-lab-imported-scopes").innerHTML = importedScopes.length
    ? importedScopes.map((row) => lessonScopeCard(row, "imported_final_asset")).join("")
    : emptyState("请选择一个老师库经验版本，或确认系统已有老师库已加载。");
  summaryCards("lesson-lab-current-summary", [
    ["source", current.final_lesson_source || "-"],
    ["teacher_scope_count", current.teacher_scope_count ?? 0],
    ["final_lesson_state_json", current.final_lesson_state_json || "-"],
    ["suite_summary_json", current.suite_summary_json || "-"],
  ]);
  $("lesson-lab-current-scopes").innerHTML = currentScopes.length
    ? currentScopes.map((row) => lessonScopeCard(row, current.final_lesson_source)).join("")
    : emptyState("当前 workflow 尚未生成 final_lesson_set。请运行 inner loop，或在 scoring 时选择系统已有老师库。");
}

function latestScoringArtifacts() {
  const files = state.provenance?.artifact_files || [];
  const scoringFiles = files.filter((path) => String(path).includes("/scoring/") && String(path).endsWith(".json"));
  const byKind = {};
  for (const path of scoringFiles) {
    const name = safeArtifactName(path);
    if (name.includes("scoring_provenance")) byKind.scoring_provenance = path;
    else if (name.includes("signal_input_manifest")) byKind.signal_input_manifest = path;
    else if (name.includes("scoring_prompt_preview")) byKind.scoring_prompt_preview = path;
    else if (name.includes("score")) byKind.scoring_result = path;
  }
  return { scoringFiles, byKind };
}

async function loadLatestScoringProvenance() {
  state.latestScoringProvenance = null;
  const path = latestScoringArtifacts().byKind.scoring_provenance || "";
  if (!path) return;
  try {
    const payload = await fetchJson(`/console/artifact-json?path=${encodeURIComponent(path)}`);
    state.latestScoringProvenance = payload.payload || null;
  } catch (error) {
    state.latestScoringProvenance = { load_error: error.message, artifact_path: path };
  }
}

function renderScoringLab() {
  const artifacts = latestScoringArtifacts();
  const result = state.simpleScoringResult || {};
  const provenance = result.scoring_provenance || state.latestScoringProvenance || {};
  summaryCards("scoring-lab-summary", [
    ["artifact_count", artifacts.scoringFiles.length],
    ["latest_result_type", result.result_type || "-"],
    ["model_called", provenance.model_called === false ? "false" : provenance.model_called ? "true" : "-"],
    ["result_valid_for_research", provenance.result_valid_for_research === false ? "false" : provenance.result_valid_for_research ? "true" : "-"],
    ["lesson_source", provenance.lesson_source || result.bundle_meta?.lesson_source || "-"],
    ["teacher_source", provenance.teacher_source || result.bundle_meta?.teacher_source || "-"],
    ["fallback_used", provenance.fallback_used ? "yes" : "no"],
    ["fallback_reason", provenance.fallback_reason || "none"],
    ["imported_final_asset", provenance.imported_final_asset ? "yes" : "no"],
    ["current_workflow_asset", provenance.current_workflow_asset ? "yes" : "no"],
    ["demo_asset", provenance.demo_asset ? "yes" : "no"],
  ]);
  $("scoring-lab-artifacts").innerHTML = artifacts.scoringFiles.length
    ? artifacts.scoringFiles
        .slice(-12)
        .reverse()
        .map((path) => artifactCard("scoring artifact", path, provenance))
        .join("")
    : emptyState("当前 run 尚未产生 scoring artifacts。请在 Simple Mode 上传结构化候选信号，或在 Scoring Lab 使用 prompt_only / dry_run。");
  $("scoring-lab-raw").textContent = result.result_type
    ? pretty(result)
    : pretty({ scoring_provenance: provenance, artifact_files: artifacts.scoringFiles, note: "No in-memory scoring result; inspect artifact paths above." });
  wireArtifactButtons();
}

function renderAuditTrail() {
  const payload = state.provenance || {};
  const contract = payload.contract || state.projectView || {};
  const chain = [
    ["user_goal", state.taskState?.original_user_message_zh || state.taskIntake?.user_request || "-", ""],
    ["task_intake", state.taskState?.task_type || state.taskIntake?.task_type || "not generated", state.taskState?.task_state_json || ""],
    ["dataset_manifest", payload.dataset_manifest_json || state.datasetManifest?.dataset_manifest_json || state.projectView?.dataset_manifest_json || "", ""],
    ["run_spec", payload.run_spec_json || "", ""],
    ["research_campaign", payload.research_campaign_json || "", ""],
    ["workflow_result", payload.workflow_result_json || "", ""],
    ["teacher_selection", payload.teacher_selection_summary_json || "", state.teacherZoo?.fallback_reason || ""],
    ["final_lesson_set", state.lessonSet?.final_lesson_state_json || "", ""],
    ["scoring_result", latestScoringArtifacts().byKind.scoring_provenance || "", ""],
  ];
  summaryCards("audit-summary", [
    ["project_id", contract.project_id || "-"],
    ["dataset_id", contract.dataset_id || "-"],
    ["run_id", contract.run_id || contract.draft_run_id || "-"],
    ["workflow_status", payload.workflow_status || state.runMonitor?.workflow_status || "-"],
    ["artifact_files", payload.artifact_files?.length ?? 0],
    ["fallback_reason", state.teacherZoo?.fallback_reason || "none"],
  ]);
  $("audit-chain").innerHTML = chain
    .map(([label, pathOrValue, note]) => {
      const isJsonPath = String(pathOrValue || "").endsWith(".json");
      return `
        <article class="timeline-card ${note ? "fallback-card" : ""}">
          <div class="card-topline">
            <span class="card-tag">${escapeHtml(label)}</span>
            <span class="card-tag">${pathOrValue && pathOrValue !== "-" ? "available" : "not generated"}</span>
          </div>
          <h3>${escapeHtml(label.replaceAll("_", " "))}</h3>
          <p>${escapeHtml(pathOrValue || "not generated")}</p>
          ${note ? `<p class="card-footnote">${escapeHtml(note)}</p>` : ""}
          ${isJsonPath ? `<button class="ghost-button open-artifact-json" type="button" data-artifact-output="audit-raw" data-artifact-path="${escapeHtml(pathOrValue)}">Open raw JSON</button>` : ""}
        </article>
      `;
    })
    .join("");
  $("audit-paths").innerHTML = (payload.artifact_files || []).length
    ? (payload.artifact_files || []).map((path) => pathCard("artifact", path)).join("")
    : emptyState("当前 run 尚未发现 artifact files。请先生成 run_spec、启动 workflow，或完成一次 scoring。");
  wireArtifactButtons();
}

function renderExpertLabs() {
  renderDatasetLab();
  renderWorkflowLab();
  renderTeacherZooLab();
  renderLessonLab();
  renderScoringLab();
  renderAuditTrail();
  renderExpertWorkbench();
}

function wireArtifactButtons() {
  document.querySelectorAll(".open-artifact-json").forEach((button) => {
    if (button.dataset.wired === "true") return;
    button.dataset.wired = "true";
    button.addEventListener("click", async () => {
      const path = button.dataset.artifactPath || "";
      const outputId = button.dataset.artifactOutput || "scoring-lab-raw";
      try {
        const payload = await fetchJson(`/console/artifact-json?path=${encodeURIComponent(path)}`);
        $(outputId).textContent = pretty(payload);
        if (outputId === "audit-raw") setWorkspace("audit-trail");
        else setWorkspace("scoring");
      } catch (error) {
        $(outputId).textContent = `Cannot open artifact JSON: ${error.message}`;
      }
    });
  });
}

function renderWizard() {
  const payload = state.wizardBundle || {};
  summaryCards("wizard-summary-cards", [
    ["Task Type", payload.run_spec?.task_type || "-"],
    ["Mode", payload.run_spec?.mode || "-"],
    ["Launchable", payload.launchable ? "yes" : "no"],
    ["Run ID", payload.run_spec?.run_id || "-"],
  ]);
  $("wizard-summary-text").textContent = payload.natural_language_summary || "No wizard artifacts yet.";
  $("wizard-plan-grid").innerHTML = (payload.pipeline_plan?.steps || []).length
    ? (payload.pipeline_plan.steps || [])
        .map(
          (step) => `
            <article class="timeline-card">
              <div class="card-topline">
                <span class="card-tag">${escapeHtml(step.step_id)}</span>
                <span class="card-tag">${escapeHtml(step.owner)}</span>
              </div>
              <h3>${escapeHtml(step.title)}</h3>
              <p>${escapeHtml(step.description || "")}</p>
              <div class="pill-row">
                ${step.wrapper_stage ? `<span class="pill">${escapeHtml(step.wrapper_stage)}</span>` : ""}
                <span class="pill">${escapeHtml(step.stage_type || "")}</span>
              </div>
            </article>
          `,
        )
        .join("")
    : emptyState("No wizard plan yet.");
  $("wizard-paths").innerHTML = payload.paths
    ? Object.entries(payload.paths)
        .map(
          ([label, value]) => `
            <div class="path-card">
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(String(value || "-"))}</strong>
            </div>
          `,
        )
        .join("")
    : emptyState("No artifact paths yet.");

  $("wizard-project-config-json").value = payload.project_config ? pretty(payload.project_config) : "";
  $("wizard-dataset-manifest-json").value = payload.dataset_manifest ? pretty(payload.dataset_manifest) : "";
  $("wizard-run-spec-json").value = payload.run_spec ? pretty(payload.run_spec) : "";
  $("wizard-campaign-json").value = payload.research_campaign ? pretty(payload.research_campaign) : "";

  $("wizard-launch").disabled = !payload.launchable || !["full_pipeline", "outer_loop_only", "inner_loop_only", "scoring_only"].includes(payload.run_spec?.mode || "");
}

function renderProjectView() {
  const payload = state.projectView || {};
  summaryCards("project-summary-cards", [
    ["Project", payload.project_id || "-"],
    ["Dataset", payload.dataset_id || "-"],
    ["Draft Run", payload.draft_run_id || "-"],
    ["Isolation", payload.data_isolation?.isolated_from_imported_assets ? "Isolated" : "Check"],
    ["Imported Fallback", payload.allow_imported_fallback ? "Allowed" : "Blocked"],
    ["Demo Fallback", payload.allow_demo_fallback ? "Allowed" : "Blocked"],
  ]);
  $("contract-path-list").innerHTML = [
    ["Dataset Root", payload.dataset_root],
    ["Dataset Raw Root", payload.dataset_raw_root],
    ["Stock Kline Root", payload.dataset_stock_klines_root],
    ["Index Kline Root", payload.dataset_index_klines_root],
    ["Dataset Cache Root", payload.dataset_cache_root],
    ["Dataset Jobs Root", payload.dataset_jobs_root],
    ["Dataset Upload Root", payload.dataset_upload_root],
    ["Asset Root", payload.asset_root],
    ["Teacher Zoo Root", payload.teacher_zoo_root],
    ["Lesson Root", payload.lesson_root],
    ["Scoring Root", payload.scoring_root],
    ["Workflow Root", payload.workflow_root],
    ["Shared Context Root", payload.shared_context_root],
    ["Imported Asset Root", payload.imported_asset_root],
    ["project_config.json", payload.project_config_json],
    ["dataset_manifest.json", payload.dataset_manifest_json],
  ]
    .map(
      ([label, value]) => `
        <div class="path-card">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value || "-")}</strong>
        </div>
      `,
    )
    .join("");
}

function renderRunSpecs() {
  $("run-spec-table-body").innerHTML = state.runSpecs
    .map((row) => {
      const isActive =
        state.activeRun &&
        row.project_id === state.activeRun.project_id &&
        row.dataset_id === state.activeRun.dataset_id &&
        row.run_id === state.activeRun.run_id;
      const fallbackText = `imported=${row.allow_imported_fallback ? "on" : "off"} · demo=${row.allow_demo_fallback ? "on" : "off"}`;
      return `
        <tr>
          <td>${escapeHtml(row.project_id)}</td>
          <td>${escapeHtml(row.dataset_id)}</td>
          <td>${escapeHtml(row.run_id)}</td>
          <td>${escapeHtml(row.mode || "-")}</td>
          <td>${escapeHtml(row.status || "-")}</td>
          <td>${escapeHtml(fallbackText)}</td>
          <td><button class="use-button" data-project="${escapeHtml(row.project_id)}" data-dataset="${escapeHtml(row.dataset_id)}" data-run="${escapeHtml(row.run_id)}">${isActive ? "Active" : "Use"}</button></td>
        </tr>
      `;
    })
    .join("");

  document.querySelectorAll("#run-spec-table-body .use-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const next = {
        project_id: button.dataset.project,
        dataset_id: button.dataset.dataset,
        run_id: button.dataset.run,
      };
      state.activeRun = next;
      applyContextToForms({ ...captureEntryContext(), ...next });
      await loadProjectView(captureEntryContext());
      await loadRunScopedViews();
      renderStatusRibbon();
      renderRunSpecs();
      setWorkspace("full-pipeline");
    });
  });
}

function renderRunSpecOutput() {
  const payload = state.latestRunSpec || {};
  const runSpec = payload.run_spec || {};
  const plan = payload.pipeline_plan || {};
  summaryCards("runspec-summary-cards", [
    ["Mode", runSpec.mode || "-"],
    ["Run ID", runSpec.run_id || "-"],
    ["Run Spec JSON", payload.run_spec_json || "-"],
    ["Research Campaign JSON", payload.research_campaign_json || "-"],
  ]);
  $("runspec-plan-grid").innerHTML = (plan.steps || [])
    .map(
      (step) => `
        <article class="timeline-card">
          <div class="card-topline">
            <span class="card-tag">${escapeHtml(step.step_id)}</span>
            <span class="card-tag">${escapeHtml(step.owner)}</span>
          </div>
          <h3>${escapeHtml(step.title)}</h3>
          <p>${escapeHtml(step.description || "")}</p>
          <div class="pill-row">
            ${step.wrapper_stage ? `<span class="pill">${escapeHtml(step.wrapper_stage)}</span>` : ""}
            <span class="pill">${escapeHtml(step.stage_type || "")}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderRunMonitor() {
  const payload = state.runMonitor || {};
  const chip = $("monitor-status-chip");
  chip.textContent = payload.workflow_status || "No Run Loaded";
  chip.classList.toggle("warn", !["completed", "partial"].includes(payload.workflow_status));

  $("monitor-strip").innerHTML = [
    ["Executed", payload.executed_steps ?? "-"],
    ["Manual", payload.manual_steps ?? "-"],
    ["Failed", payload.failed_steps ?? "-"],
    ["Workflow Result", payload.workflow_result_json || "-"],
  ]
    .map(
      ([label, value]) => `
        <div class="kv">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(String(value ?? "-"))}</strong>
        </div>
      `,
    )
    .join("");

  $("monitor-grid").innerHTML = (payload.nodes || []).length
    ? (payload.nodes || [])
        .map(
          (node) => `
            <article class="timeline-card ${node.status === "failed" ? "warn-card" : ""}">
              <div class="card-topline">
                <span class="card-tag">${escapeHtml(node.node_id)}</span>
                <span class="card-tag">${escapeHtml(node.status)}</span>
              </div>
              <h3>${escapeHtml(node.label)}</h3>
              <p>${escapeHtml(node.payload?.artifact_json || node.payload?.suite_summary_json || "No artifact yet.")}</p>
            </article>
          `,
        )
        .join("")
    : emptyState("还没有 workflow_result。请先生成 run_spec 并启动 workflow；如果只是评分，可先使用系统已有老师库。");
}

function renderTeacherZoo() {
  const payload = state.teacherZoo || {};
  const importedTeachers = (payload.imported_teachers || []).length ? payload.imported_teachers || [] : state.importedTeachers || [];
  summaryCards("teacher-zoo-summary", [
    ["Selection Source", payload.selection_resolution_source || "-"],
    ["Fallback Reason", payload.fallback_reason || "none"],
    ["System Library Teachers", importedTeachers.length ?? 0],
    ["Selected For Inner Loop", payload.selected_teachers_for_inner_loop?.length ?? 0],
  ]);

  $("imported-teacher-list").innerHTML = importedTeachers.length
    ? importedTeachers.map(teacherCard).join("")
    : emptyState("当前 profile 暂未发现系统已有老师。");
  $("selected-teacher-list").innerHTML = (payload.selected_teachers_for_inner_loop || []).length
    ? (payload.selected_teachers_for_inner_loop || []).map(teacherCard).join("")
    : emptyState("This run has not produced selected inner-loop teachers yet.");
  $("candidate-teacher-list").innerHTML = (payload.current_workflow_candidate_teachers || []).length
    ? (payload.current_workflow_candidate_teachers || []).map(teacherCard).join("")
    : emptyState("No current-workflow candidate teachers. This is expected for scoring-only or spec-only runs.");
  $("validated-teacher-list").innerHTML = (payload.current_workflow_validated_teachers || []).length
    ? (payload.current_workflow_validated_teachers || []).map(teacherCard).join("")
    : emptyState("No validated teachers are attached to this run.");
  $("frozen-teacher-list").innerHTML = (payload.current_workflow_frozen_teachers || []).length
    ? (payload.current_workflow_frozen_teachers || []).map(teacherCard).join("")
    : emptyState("No frozen-eval teachers are attached to this run.");
}

function renderLessonSet() {
  const payload = state.lessonSet || {};
  summaryCards("lesson-set-summary", [
    ["Final Lesson Source", payload.final_lesson_source || "-"],
    ["Final Lesson JSON", payload.final_lesson_state_json || "-"],
    ["Suite Summary JSON", payload.suite_summary_json || "-"],
    ["Teacher Scopes", payload.teacher_scope_count ?? 0],
  ]);

  $("lesson-scope-grid").innerHTML = (payload.teacher_scopes || []).length
    ? (payload.teacher_scopes || [])
        .map(
          (row) => `
            <article class="lesson-card">
              <div class="card-topline">
                <span class="card-tag">${escapeHtml(row.round_id)}</span>
                ${sourcePill(row.source_type)}
              </div>
              <h3>${escapeHtml(row.lesson_name || "Untitled Lesson")}</h3>
              <p>source_round_id=${escapeHtml(row.source_round_id || "-")}</p>
              <div class="pill-row">
                <span class="pill">${escapeHtml(String(row.item_count))} items</span>
                <span class="pill">${escapeHtml(String(row.meta_rule_count))} meta rules</span>
              </div>
            </article>
          `,
        )
        .join("")
    : emptyState("This run does not have a current workflow final lesson set yet.");
}

function renderProvenance() {
  const payload = state.provenance || {};
  summaryCards("provenance-summary", [
    ["project_config.json", payload.project_config_json || "-"],
    ["dataset_manifest.json", payload.dataset_manifest_json || "-"],
    ["Run Spec", payload.run_spec_json || "-"],
    ["Workflow Result", payload.workflow_result_json || "-"],
  ]);

  $("provenance-paths").innerHTML = (payload.artifact_files || []).length
    ? (payload.artifact_files || [])
        .map(
          (path) => `
            <div class="path-card">
              <span>artifact</span>
              <strong>${escapeHtml(path)}</strong>
            </div>
          `,
        )
        .join("")
    : emptyState("还没有 provenance files。请先生成 run_spec、启动 workflow，或完成一次 scoring。");
}

function populateStaticSelectors() {
  const lessonOptions = state.lessons
    .map((item) => `<option value="${escapeHtml(item.alias)}">${escapeHtml(item.alias)} · ${escapeHtml(item.seed_label)}</option>`)
    .join("");
  $("scoring-lesson-alias").innerHTML = lessonOptions;
  if ($("library-lesson-alias")) $("library-lesson-alias").innerHTML = lessonOptions;
  if ($("lesson-lab-imported-alias")) $("lesson-lab-imported-alias").innerHTML = lessonOptions;
  $("scoring-recorded-run").innerHTML = state.markets
    .map((item) => `<option value="${escapeHtml(item.alias)}">${escapeHtml(item.alias)} · ${escapeHtml(item.window)}</option>`)
    .join("");
  populateSimpleScoringSelectors();
}

function renderScoringSingle(payload) {
  const live = payload || {};
  const meta = live.bundle_meta || {};
  const teacherCards = meta.teacher_cards || [];
  const teacherScores = live.teacher_scores || live.parsed_payload?.teacher_scores || [];
  summaryCards("scoring-result-summary", [
    ["Teacher Library", meta.teacher_library_name_zh || teacherLibraryName(meta.teacher_library_id || "")],
    ["Lesson Source", sourceLabelZh(meta.lesson_source || "-")],
    ["Teacher Source", sourceLabelZh(meta.teacher_source || "-")],
    ["Fallback Used", meta.fallback_used ? "yes" : "no"],
    ["Fallback Reason", meta.fallback_reason || "none"],
    ["System Teacher Library", meta.lesson_source === "imported_final_asset" ? "yes" : "no"],
    ["Current Workflow Final Lesson", meta.lesson_source === "current_workflow_asset" ? "yes" : "no"],
  ]);
  $("scoring-total-score").textContent = live.mode === "prompt_only" ? "Prompt" : live.total_score ?? "-";
  $("scoring-short-reason").textContent =
    live.mode === "prompt_only" ? "Prompt only mode: no model call was made." : live.short_reason || "No short reason returned.";
  $("scoring-teacher-breakdown").innerHTML = teacherCards.length
    ? teacherCards
        .map((teacher) => {
          const matched = teacherScores.find((row) => row.round_id === teacher.round_id) || {};
          const scoreLabel = live.mode === "prompt_only" ? "prompt only" : matched.score ?? "-";
          return `
            <article class="teacher-score-card">
              <div class="card-topline">
                <span class="card-tag">${escapeHtml(teacher.round_id || "-")}</span>
                ${sourcePill(teacher.source_type)}
              </div>
              <h3>${escapeHtml(teacher.title || teacher.style_family || "Teacher")}</h3>
              <strong>${escapeHtml(String(scoreLabel))}</strong>
              <p>${escapeHtml(matched.note || teacher.style_family || teacher.basic_filter || "Teacher-local compatibility score.")}</p>
            </article>
          `;
        })
        .join("")
    : emptyState("No teacher-level breakdown was returned.");
  $("scoring-raw").textContent = pretty(live);
  $("scoring-batch-body").innerHTML = "";
}

function renderScoringBatch(payload) {
  const summary = payload.summary || {};
  summaryCards("scoring-result-summary", [
    ["Batch Count", summary.count ?? 0],
    ["Imported Lesson", summary.used_imported_lesson ? "yes" : "no"],
    ["Current Workflow Final Lesson", summary.used_current_workflow_final_lesson_set ? "yes" : "no"],
    ["Profile", summary.profile || "-"],
    ["Lesson Alias", summary.lesson_alias || "-"],
    ["Prompt Only", summary.prompt_only ? "yes" : "no"],
  ]);
  $("scoring-total-score").textContent = summary.count ?? 0;
  $("scoring-short-reason").textContent = "Batch scoring completed. Inspect item-level provenance below.";
  $("scoring-teacher-breakdown").innerHTML = emptyState("Teacher-by-teacher numeric breakdown is shown for single-signal scoring. Batch mode summarizes provenance item by item.");
  $("scoring-raw").textContent = pretty(payload);
  $("scoring-batch-body").innerHTML = (payload.items || [])
    .map((item, idx) => {
      const meta = item.bundle_meta || {};
      const signal = item.signal_schema_validation?.reference_signal || item.parsed_payload?.signal || {};
      return `
        <tr>
          <td>${idx + 1}</td>
          <td>${escapeHtml(signal.symbol || item.signal_record?.symbol || "-")}</td>
          <td>${escapeHtml(signal.signal_date || item.signal_record?.signal_date || "-")}</td>
          <td>${escapeHtml(String(item.total_score ?? "-"))}</td>
          <td>${escapeHtml(meta.lesson_source || "-")}</td>
          <td>${escapeHtml(meta.teacher_source || "-")}</td>
          <td>${escapeHtml(meta.fallback_used ? "yes" : "no")}</td>
        </tr>
      `;
    })
    .join("");
}

function populateSimpleScoringSelectors() {
  const lessonOptions = state.lessons
    .map((item) => `<option value="${escapeHtml(item.alias)}">${escapeHtml(item.alias)} · ${escapeHtml(item.seed_label)}</option>`)
    .join("");
  if ($("simple-scoring-lesson-alias")) $("simple-scoring-lesson-alias").innerHTML = lessonOptions;
  const marketOptions = state.markets
    .map((item) => `<option value="${escapeHtml(item.alias)}">${escapeHtml(item.alias)} · ${escapeHtml(item.window)}</option>`)
    .join("");
  if ($("simple-scoring-schema-run")) $("simple-scoring-schema-run").innerHTML = marketOptions;
}

function renderSimpleScoringResult(payload = state.simpleScoringResult) {
  if (!$("simple-scoring-summary")) return;
  const result = payload || {};
  if (!result.mode) {
    $("simple-scoring-summary").innerHTML = emptyState("还没有评分结果。你可以直接输入股票代码让我判断需要哪些数据，或上传结构化候选信号后进行评分。");
    $("simple-scoring-teacher-breakdown").innerHTML = "";
    $("simple-scoring-raw").textContent = "还没有评分产物。";
    return;
  }
  const provenance = result.scoring_provenance || {};
  const manifest = result.signal_input_manifest || {};
  const paths = result.artifact_paths || {};
  const live = result.mode === "live";
  const noModelText = live ? "live scoring" : "这是调试 / 校验模式，不是 live scoring";
  const savedRaw = Boolean(paths.live_cache_json || provenance.cache_path || provenance.raw_response_available);
  const savedParsed = Boolean(paths.live_saved_run_json || provenance.saved_run_path || paths.live_cache_json);
  const score60 = result.score_60d ?? result.window_scores?.score_60d ?? result.parsed_payload?.score_60d;
  const score120 = result.score_120d ?? result.window_scores?.score_120d ?? result.parsed_payload?.score_120d;
  const windowScoreNote = result.window_score_note || result.parsed_payload?.window_score_note || "";
  const diagnostics = result.feature_diagnostics_zh || {};
  const featureCues = diagnostics.cues_zh || [];
  const riskFlags = diagnostics.risk_flags_zh || [];
  const tradingReference = result.trading_reference_zh || "";
  const primaryReason = tradingReference || (isMostlyChinese(result.short_reason) ? result.short_reason : result.summary_zh) || (live ? "评分已完成。" : "本模式只用于输入校验或预览。");
  const provenanceLessonSource = provenance.lesson_source || result.bundle_meta?.lesson_source || "";
  const teacherLibraryId = result.teacher_library_id
    || result.teacher_library?.teacher_library_id
    || provenance.teacher_library_id
    || result.bundle_meta?.teacher_library_id
    || "";
  const teacherLibraryDisplay = result.teacher_library_name_zh
    || result.teacher_library?.display_name_zh
    || provenance.teacher_library_name_zh
    || result.bundle_meta?.teacher_library_name_zh
    || teacherLibraryName(teacherLibraryId);
  $("simple-scoring-summary").innerHTML = `
    <article class="summary-card result-primary-card">
      <span class="metric-label">主结果</span>
      <strong>${escapeHtml(live ? `${result.total_score ?? "-"} / 100` : noModelText)}</strong>
      <p>${escapeHtml(primaryReason)}</p>
    </article>
    <article class="summary-card ${live && score60 !== undefined ? "ok-card" : "warn-card"}">
      <span class="metric-label">近 60 日评分</span>
      <strong>${escapeHtml(score60 !== undefined ? `${score60} / 100` : "待窗口化特征")}</strong>
      <p>${escapeHtml(score60 !== undefined ? "偏短线，强调近 60 交易日的动量、波动与位置结构。" : "需要 live 返回 score_60d，或输入包含 60 日窗口特征的结构化 signal。")}</p>
    </article>
    <article class="summary-card ${live && score120 !== undefined ? "ok-card" : "warn-card"}">
      <span class="metric-label">近 120 日评分</span>
      <strong>${escapeHtml(score120 !== undefined ? `${score120} / 100` : "待窗口化特征")}</strong>
      <p>${escapeHtml(score120 !== undefined ? "偏中期，强调近 120 交易日的趋势背景与回撤位置。" : "普通模式默认只拉取近 120 交易日上下文；后续窗口化因子会写入这里。")}</p>
    </article>
    <article class="summary-card ${live ? "ok-card" : "warn-card"}">
      <span class="metric-label">模式</span>
      <strong>${escapeHtml(live ? "本地模型实时评分" : "预览 / 校验模式")}</strong>
      <p>${escapeHtml(live ? `已调用本地 GPT-OSS；结果仅用于研究辅助。${windowScoreNote && isMostlyChinese(windowScoreNote) ? ` ${windowScoreNote}` : ""}` : "当前没有真实调用模型。")}</p>
    </article>
    <article class="summary-card">
      <span class="metric-label">模型调用</span>
      <strong>${escapeHtml(provenance.model_called ? "已调用本地模型" : "未调用模型")}</strong>
      <p>${escapeHtml(live ? "模型：本地 GPT-OSS" : "可切换到 live，但需要本地 runtime 健康。")}</p>
    </article>
    <article class="summary-card">
      <span class="metric-label">使用老师库</span>
      <strong>${escapeHtml(teacherLibraryDisplay)}</strong>
      <p>${escapeHtml(provenance.fallback_used ? `发生回退：${provenance.fallback_reason || "使用备用经验规则"}` : `来源：${sourceLabelZh(provenanceLessonSource)}。`)}</p>
    </article>
    <article class="summary-card result-primary-card">
      <span class="metric-label">技术结构</span>
      <strong>${escapeHtml(featureCues.length ? "已生成因子诊断" : "等待因子诊断")}</strong>
      <p>${featureCues.length ? featureCues.slice(0, 4).map((item) => `• ${item}`).join("\n") : "live 单股评分会展示 MA10/MA20、KDJ、pos_20、量比、波动率、收益率结构等专有名词诊断。"}</p>
    </article>
    <article class="summary-card ${riskFlags.length ? "warn-card" : "ok-card"}">
      <span class="metric-label">参考意见</span>
      <strong>${escapeHtml(tradingReference ? "可参考" : "等待生成")}</strong>
      <p>${escapeHtml(tradingReference || (riskFlags.length ? riskFlags.join("；") : "暂无单独参考意见。"))}</p>
    </article>
    <article class="summary-card warn-card">
      <span class="metric-label">风险边界</span>
      <strong>不构成投资建议</strong>
      <p>该结果仅用于研究辅助，不保证收益，也不应直接作为交易建议。</p>
    </article>
  `;
  const teachers = result.teacher_scores || result.bundle_meta?.teacher_cards || [];
  $("simple-scoring-teacher-breakdown").innerHTML = teachers.length
    ? teachers
        .map((teacher) => `
          <article class="teacher-score-card">
            <div class="card-topline">
              <span class="card-tag">老师模型</span>
            </div>
            <h3>${escapeHtml(teacherDisplayNameZh(teacher))}</h3>
            <strong>${escapeHtml(String(teacher.score ?? (result.mode === "mock" ? "-" : "preview only")))}</strong>
            <p>${escapeHtml(teacherNoteZh(teacher))}</p>
            <details class="details-block compact">
              <summary>技术标识</summary>
              <code>${escapeHtml(teacher.round_id || "-")}</code>
            </details>
          </article>
        `)
        .join("")
    : emptyState("暂无 teacher-by-teacher 分数。prompt_only / dry_run 只生成输入校验与 prompt preview。");
  $("simple-scoring-raw").textContent = result.mode
    ? pretty(result)
    : "还没有 Simple scoring 结果。";
}

async function collectSimpleScoringInput() {
  const mode = $("simple-scoring-mode")?.value || "prompt_only";
  if (mode === "live" && !isLiveRuntimeHealthy()) {
    throw new Error("当前本地 GPT-OSS runtime 未健康，不能执行 live scoring。请先启动本地模型，或改用提示词预览、输入校验、mock 或历史样本复核。");
  }
  const file = $("simple-scoring-file")?.files?.[0] || null;
  const filePayload = file ? await readFilePayload(file) : null;
  const pasted = $("simple-scoring-json")?.value?.trim() || "";
  const scoringPayload = {
    mode,
    lesson_source_mode: $("simple-scoring-source")?.value || "auto",
    lesson_alias: $("simple-scoring-lesson-alias")?.value || "alignment_seed0005",
    schema_market_run_alias: $("simple-scoring-schema-run")?.value || "",
    allow_lesson_fallback: Boolean($("simple-scoring-allow-fallback")?.checked),
    pasted_payload: pasted,
    archived_run: $("simple-scoring-schema-run")?.value || "",
    archived_signal_date: $("simple-scoring-archived-date")?.value?.trim() || "",
    archived_symbol: $("simple-scoring-archived-symbol")?.value?.trim() || "",
  };
  if (!filePayload && !pasted && mode !== "archived_replay") {
    throw new Error("请上传结构化 signal 文件，或粘贴 signal JSON。只输入股票代码不能评分。");
  }
  return { filePayload, scoringPayload };
}

function parseLiveSignalRecords(filePayload, pastedPayload) {
  if (filePayload && filePayload.content_encoding !== "text") {
    throw new Error("live scoring 当前只接受 JSON 文本信号。Parquet/二进制文件可先用 dry_run 校验，或转换为 JSON signal record。");
  }
  const raw = String((filePayload && filePayload.content) || pastedPayload || "").trim();
  if (!raw) {
    throw new Error("live scoring 需要结构化 JSON signal record，不能只输入股票代码或自然语言。");
  }
  const parsed = JSON.parse(raw);
  const records = Array.isArray(parsed) ? parsed : [parsed];
  return records.map(normalizeSignalDateAlias);
}

function normalizeSignalDateAlias(record) {
  const row = { ...(record || {}) };
  if (!row.signal_date && row.date) {
    row.signal_date = row.date;
    row._studio_normalization_note = "date was accepted as an alias and normalized to signal_date.";
  }
  return row;
}

function decorateLiveScoringResult(payload, recordCount) {
  const isBatch = Boolean(payload.summary && payload.items);
  const first = isBatch ? (payload.items || [])[0] || {} : payload;
  const meta = first.bundle_meta || payload.bundle_meta || {};
  const validation = first.signal_schema_validation || payload.signal_schema_validation || {};
  const provenance = {
    created_at: new Date().toISOString(),
    mode: "live",
    model_called: isBatch ? (payload.items || []).some((item) => item.model_called || item.mode === "live_runtime") : Boolean(payload.model_called || payload.mode === "live_runtime"),
    result_valid_for_research: isBatch ? (payload.items || []).every((item) => item.result_valid_for_research !== false && item.total_score !== undefined) : Boolean(payload.result_valid_for_research),
    lesson_source: meta.lesson_source || "-",
    teacher_source: meta.teacher_source || "-",
    fallback_used: Boolean(meta.fallback_used),
    fallback_reason: meta.fallback_reason || "",
    imported_final_asset: meta.lesson_source === "imported_final_asset" || meta.teacher_source === "imported_final_asset",
    current_workflow_asset: meta.lesson_source === "current_workflow_asset" || meta.teacher_source === "current_workflow_asset",
    demo_asset: meta.lesson_source === "demo_asset" || meta.teacher_source === "demo_asset",
    vllm_started: true,
    external_api_called: false,
    local_runtime_used: true,
    cache_path: first.cache_path || payload.cache_path || "",
    saved_run_path: first.saved_run_path || payload.saved_run_path || "",
    raw_response_available: Boolean(first.raw_response || payload.raw_response),
  };
  return {
    ...payload,
    mode: "live",
    result_type: isBatch ? "live_model_batch_scoring" : "live_model_scoring",
    scoring_provenance: provenance,
    signal_input_manifest: {
      valid: validation.valid !== false,
      record_count: recordCount,
      missing_columns: validation.missing_top_level_keys || validation.missing_columns || [],
      missing_feature_keys: validation.missing_feature_keys || [],
      non_numeric_feature_keys: validation.non_numeric_feature_keys || [],
    },
    artifact_paths: {
      live_cache_json: first.cache_path || payload.cache_path || "",
      live_saved_run_json: first.saved_run_path || payload.saved_run_path || "",
    },
  };
}

function updateTaskStateFromSimpleScoring(result) {
  if (!state.taskState || !result) return;
  const provenance = result.scoring_provenance || {};
  const manifest = result.signal_input_manifest || {};
  const paths = result.artifact_paths || {};
  state.taskState.current_stage = "scoring_artifact_ready";
  state.taskState.scoring_status = manifest.valid === false ? "validation_failed" : "completed";
  state.taskState.scoring_mode = result.mode || "";
  state.taskState.model_called = Boolean(provenance.model_called);
  state.taskState.result_valid_for_research = Boolean(provenance.result_valid_for_research);
  state.taskState.scored_signal_count = manifest.record_count || 0;
  state.taskState.lesson_source = provenance.lesson_source || "";
  state.taskState.teacher_source = provenance.teacher_source || "";
  state.taskState.fallback_used = Boolean(provenance.fallback_used);
  state.taskState.fallback_reason = provenance.fallback_reason || "";
  state.taskState.last_score = result.total_score ?? result.summary?.mean_score ?? "";
  state.taskState.last_score_60d = result.score_60d ?? result.parsed_payload?.score_60d ?? "";
  state.taskState.last_score_120d = result.score_120d ?? result.parsed_payload?.score_120d ?? "";
  state.taskState.last_scoring_result_path = paths.live_saved_run_json || paths.live_cache_json || paths["scoring_provenance.json"] || "";
  state.taskState.next_action = "open_expert_monitor";
}

async function runSimpleScoringFlow() {
  const { filePayload, scoringPayload } = await collectSimpleScoringInput();
  if (scoringPayload.mode === "live") {
    const confirmed = window.confirm("这将调用本地 GPT-OSS 进行 live scoring，不是 dry_run。结果仅用于研究辅助，不构成投资建议。是否继续？");
    if (!confirmed) {
      state.chatMessages.push({
        role: "assistant",
        content: "已取消 live scoring。本地模型没有被调用。你可以改用 prompt_only 或 dry_run 先校验输入。",
      });
      renderSimpleChatThread();
      return null;
    }
    const signalRecords = parseLiveSignalRecords(filePayload, scoringPayload.pasted_payload);
    const sourceMode = scoringPayload.lesson_source_mode || "auto";
    const useCurrent = sourceMode === "current_workflow" || (sourceMode === "auto" && state.lessonSet?.final_lesson_state_json);
    if (sourceMode === "current_workflow" && !state.lessonSet?.final_lesson_state_json) {
      throw new Error("当前 workflow 还没有 final_lesson_set，不能使用本次训练老师库做 live scoring。请先完成 inner loop，或选择系统已有老师库。");
    }
    const common = {
      profile: state.profile,
      lesson_alias: useCurrent ? "" : scoringPayload.lesson_alias || "alignment_seed0005",
      final_lesson_state_json: useCurrent ? state.lessonSet?.final_lesson_state_json || "" : "",
      prompt_only: false,
      reuse_cache: false,
      persist_run: true,
      run_label: `${state.taskState?.run_id || state.activeRun?.run_id || "simple"}__live_scoring`,
      schema_from_run: scoringPayload.schema_market_run_alias || "",
    };
    const payload = signalRecords.length === 1
      ? await fetchJson("/score/live", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...common, signal_record: signalRecords[0] }),
        })
      : await fetchJson("/score/live-batch-external", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...common, signal_records: signalRecords }),
    });
    state.simpleScoringResult = decorateLiveScoringResult(payload, signalRecords.length);
    updateTaskStateFromSimpleScoring(state.simpleScoringResult);
    const score60 = state.simpleScoringResult.score_60d ?? state.simpleScoringResult.parsed_payload?.score_60d ?? "-";
    const score120 = state.simpleScoringResult.score_120d ?? state.simpleScoringResult.parsed_payload?.score_120d ?? "-";
    state.chatMessages.push({
      role: "assistant",
      content: `评分已完成：综合评分 ${state.simpleScoringResult.total_score ?? "-"} / 100；近 60 日评分 ${score60} / 100；近 120 日评分 ${score120} / 100。已调用本地 GPT-OSS，共处理 ${signalRecords.length} 条信号。结果仅用于研究辅助，不构成投资建议。`,
    });
    renderSimpleScoringResult();
    renderSimpleTaskState();
    renderSimpleArtifactLinks();
    renderSimpleChatThread();
    return state.simpleScoringResult;
  }
  const payload = await executeChatAction("score_signal", {
    confirm: true,
    file_payload: filePayload,
    scoring_payload: scoringPayload,
  });
  state.simpleScoringResult = payload.scoring_result || state.simpleScoringResult;
  updateTaskStateFromSimpleScoring(state.simpleScoringResult);
  renderSimpleScoringResult();
  return payload;
}

async function loadProjectView(context = captureEntryContext()) {
  state.profile = context.profile;
  state.projectView = await fetchJson(
    `/console/project?profile=${encodeURIComponent(context.profile)}&project_id=${encodeURIComponent(context.project_id)}&dataset_id=${encodeURIComponent(context.dataset_id)}&run_id=${encodeURIComponent(context.run_id)}&allow_imported_fallback=${context.allow_imported_fallback}&allow_demo_fallback=${context.allow_demo_fallback}`,
  );
  try {
    state.datasetManifest = await fetchJson(
      `/console/dataset-manifest?profile=${encodeURIComponent(context.profile)}&project_id=${encodeURIComponent(context.project_id)}&dataset_id=${encodeURIComponent(context.dataset_id)}&run_id=${encodeURIComponent(context.run_id)}&allow_imported_fallback=${context.allow_imported_fallback}&allow_demo_fallback=${context.allow_demo_fallback}`,
    );
  } catch (error) {
    state.datasetManifest = {
      exists: false,
      load_error: error.message,
      project_id: context.project_id,
      dataset_id: context.dataset_id,
      run_id: context.run_id,
    };
  }
  applyContextToForms(state.projectView);
  renderProjectView();
  renderDatasetManifest();
  renderStatusRibbon();
  renderDashboard();
  renderExpertLabs();
}

async function loadRunSpecs() {
  const payload = await fetchJson("/console/runs");
  state.runSpecs = payload.items || [];
  renderRunSpecs();
}

async function loadRunScopedViews() {
  if (!state.activeRun) return;
  const qs = `project_id=${encodeURIComponent(state.activeRun.project_id)}&dataset_id=${encodeURIComponent(state.activeRun.dataset_id)}&run_id=${encodeURIComponent(state.activeRun.run_id)}`;
  const safeLoad = async (url, fallback) => {
    try {
      return await fetchJson(url);
    } catch (error) {
      return { ...fallback, load_error: error.message };
    }
  };
  state.runMonitor = await safeLoad(`/console/run-monitor?${qs}`, {
    workflow_status: "unavailable",
    executed_steps: "-",
    manual_steps: "-",
    failed_steps: "-",
    workflow_result_json: "",
    nodes: [],
  });
  state.teacherZoo = await safeLoad(`/console/teacher-zoo?profile=${encodeURIComponent(state.profile)}&${qs}`, {
    selection_resolution_source: "unavailable",
    fallback_reason: "run-scoped teacher zoo could not be loaded",
    imported_teachers: state.importedTeachers || [],
    selected_teachers_for_inner_loop: [],
    current_workflow_candidate_teachers: [],
    current_workflow_validated_teachers: [],
    current_workflow_frozen_teachers: [],
  });
  state.lessonSet = await safeLoad(`/console/lesson-set?${qs}`, {
    final_lesson_source: "unavailable",
    final_lesson_state_json: "",
    suite_summary_json: "",
    teacher_scope_count: 0,
    teacher_scopes: [],
  });
  state.provenance = await safeLoad(`/console/provenance?${qs}`, {
    project_config_json: "",
    dataset_manifest_json: "",
    run_spec_json: "",
    workflow_result_json: "",
    artifact_files: [],
  });
  const libraryRegistry = await safeLoad(`/teacher-libraries?profile=${encodeURIComponent(state.profile)}&${qs}`, {
    items: state.teacherLibraries || [],
    active_teacher_library_id: state.activeTeacherLibraryId,
  });
  state.teacherLibraries = libraryRegistry.items || state.teacherLibraries || [];
  state.activeTeacherLibraryId = libraryRegistry.active_teacher_library_id
    || libraryRegistry.default_teacher_library_id
    || state.teacherLibraries?.[0]?.teacher_library_id
    || state.activeTeacherLibraryId
    || "";
  await loadLatestScoringProvenance();
  renderRunMonitor();
  renderTeacherZoo();
  renderLessonSet();
  renderProvenance();
  renderExpertLabs();
  renderDashboard();
  renderSimpleMode();
}

function stopSimpleWorkflowPolling() {
  if (state.simpleWorkflowPoller) {
    clearInterval(state.simpleWorkflowPoller);
    state.simpleWorkflowPoller = null;
  }
}

async function refreshSimpleRunStatus({ appendMessage = false } = {}) {
  if (!state.activeRun?.project_id || !state.activeRun?.dataset_id || !state.activeRun?.run_id) return null;
  const qs = `profile=${encodeURIComponent(state.profile)}&project_id=${encodeURIComponent(state.activeRun.project_id)}&dataset_id=${encodeURIComponent(state.activeRun.dataset_id)}&run_id=${encodeURIComponent(state.activeRun.run_id)}`;
  const payload = await fetchJson(`/chat/run-status?${qs}`);
  state.simpleRunStatus = payload;
  state.simpleWorkflowPollErrors = 0;
  const card = payload.task_card || {};
  if (state.taskState) {
    state.taskState.current_stage = card.current_stage || state.taskState.current_stage;
    state.taskState.workflow_status = payload.workflow_status;
    state.taskState.failed_stage = card.failed_stage || "";
    state.taskState.fallback_used = card.fallback_used || false;
    state.taskState.fallback_reason = card.fallback_reason || "";
    state.taskState.latest_artifact = card.latest_artifact || "";
    state.taskState.next_action = card.next_action || state.taskState.next_action;
  }
  const signature = `${payload.workflow_status || "-"}::${card.current_stage || "-"}::${card.failed_stage || "-"}`;
  if (appendMessage && signature !== state.simpleLastWorkflowSignature) {
    state.simpleLastWorkflowSignature = signature;
    if (payload.workflow_status === "completed") {
      state.chatMessages.push({
        role: "assistant",
        content: "本次 workflow 已完成。你可以查看 Teacher Zoo、Final Lesson Set，或上传新信号进行评分；也可以进入 Expert Mode 查看 provenance。",
      });
    } else if (payload.workflow_status === "failed") {
      state.chatMessages.push({
        role: "assistant",
        content: `本次 workflow 在 ${card.failed_stage || "未知"} 阶段失败。建议查看专业日志或 Audit Trail，并检查 run_spec / 数据输入。`,
      });
    } else if (payload.workflow_status === "running" || payload.workflow_status === "pending") {
      state.chatMessages.push({
        role: "assistant",
        content: `workflow 状态：${payload.workflow_status}，当前阶段：${card.current_stage || "暂无"}。`,
      });
    }
  }
  renderSimpleMode();
  return payload;
}

function startSimpleWorkflowPolling({ appendMessage = true } = {}) {
  stopSimpleWorkflowPolling();
  state.simpleWorkflowPollErrors = 0;
  state.simpleWorkflowPoller = setInterval(async () => {
    try {
      const payload = await refreshSimpleRunStatus({ appendMessage });
      const status = payload?.workflow_status || "";
      if (["completed", "failed", "cancelled", "partial", "spec_only"].includes(status)) {
        stopSimpleWorkflowPolling();
      }
    } catch (error) {
      state.simpleWorkflowPollErrors += 1;
      if (state.simpleWorkflowPollErrors >= 3) {
        stopSimpleWorkflowPolling();
        state.chatMessages.push({
          role: "assistant",
          content: `运行状态轮询已停止：连续 ${state.simpleWorkflowPollErrors} 次请求失败。你可以打开 Expert Mode 查看详细状态。最后错误：${error.message}`,
        });
        renderSimpleChatThread();
      }
    }
  }, 4000);
}

async function refreshBaseData() {
  state.overview = await fetchJson(`/overview?profile=${encodeURIComponent(state.profile)}`);
  state.liveConfig = await fetchJson(`/live-config?profile=${encodeURIComponent(state.profile)}`);
  const context = captureEntryContext();
  const libraryRegistry = await fetchJson(
    `/teacher-libraries?profile=${encodeURIComponent(state.profile)}&project_id=${encodeURIComponent(context.project_id)}&dataset_id=${encodeURIComponent(context.dataset_id)}&run_id=${encodeURIComponent(context.run_id)}`,
  );
  state.teacherLibraries = libraryRegistry.items || [];
  state.activeTeacherLibraryId = libraryRegistry.active_teacher_library_id
    || libraryRegistry.default_teacher_library_id
    || state.teacherLibraries?.[0]?.teacher_library_id
    || "";
  state.importedTeachers = (((await fetchJson(`/teachers?profile=${encodeURIComponent(state.profile)}`)).items || []).map((row) => ({
    ...row,
    teacher_state: "built_in_frozen_teacher",
    source_type: "built_in_baseline",
  })));
  state.lessons = (await fetchJson(`/lessons?profile=${encodeURIComponent(state.profile)}`)).items || [];
  state.markets = (await fetchJson(`/markets?profile=${encodeURIComponent(state.profile)}`)).items || [];
  const defaultLessonAlias = state.lessons?.[0]?.alias || "";
  if (defaultLessonAlias) {
    state.importedLessonDetail = await fetchJson(
      `/lessons?profile=${encodeURIComponent(state.profile)}&alias=${encodeURIComponent(defaultLessonAlias)}`,
    );
  } else {
  state.importedLessonDetail = null;
  }
  renderTopMetrics();
  renderRuntimeAwareControls();
  populateStaticSelectors();
  renderDashboard();
  renderTeacherZoo();
  renderImportedLessonBrowser();
  renderExpertLabs();
}

async function loadImportedLessonDetail(alias) {
  const resolved = String(alias || "").trim();
  if (!resolved) {
    state.importedLessonDetail = null;
    renderImportedLessonBrowser();
    renderLessonLab();
    return;
  }
  state.importedLessonDetail = await fetchJson(
    `/lessons?profile=${encodeURIComponent(state.profile)}&alias=${encodeURIComponent(resolved)}`,
  );
  renderImportedLessonBrowser();
  renderLessonLab();
}

async function analyzeTaskIntake() {
  const ctx = captureEntryContext();
  const body = {
    ...ctx,
    user_request: $("intake-user-request").value.trim(),
  };
  state.taskIntake = await fetchJson("/guided/task-intake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  applyContextToForms({
    ...ctx,
    project_id: state.taskIntake.contract_preview?.project_id || ctx.project_id,
    dataset_id: state.taskIntake.contract_preview?.dataset_id || ctx.dataset_id,
    run_id: state.taskIntake.contract_preview?.run_id || ctx.run_id,
  });
  await loadProjectView(captureEntryContext());
  renderTaskIntake();
  setWorkspace("full-pipeline");
}

async function readUploadedDataset() {
  const file = $("dataset-file").files?.[0];
  if (file) {
    return readFilePayload(file);
  }
  const pasted = $("dataset-content").value.trim();
  if (pasted) {
    return {
      filename: "pasted_dataset.txt",
      content: pasted,
      content_encoding: "text",
    };
  }
  throw new Error("Please select a CSV / JSON / Parquet file or paste a dataset payload first.");
}

async function readFilePayload(file) {
  if (!file) throw new Error("请选择要上传的数据文件。");
  if (file.name.toLowerCase().endsWith(".parquet")) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return {
      filename: file.name,
      content: btoa(binary),
      content_encoding: "base64",
    };
  }
  return {
    filename: file.name,
    content: await file.text(),
    content_encoding: "text",
  };
}

function stopKlineJobPolling() {
  if (state.klineJobPoller) {
    clearInterval(state.klineJobPoller);
    state.klineJobPoller = null;
  }
}

async function validateDatasetManifest() {
  if (!state.taskIntake?.task_type) {
    throw new Error("Analyze the task first so the system knows which dataset schema to validate against.");
  }
  const ctx = captureEntryContext();
  const source = selectedDatasetSource();
  stopKlineJobPolling();

  if (source === "upload_local") {
    const dataset = await readUploadedDataset();
    state.datasetManifest = await fetchJson("/guided/dataset-onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...ctx,
        task_type: state.taskIntake.task_type,
        filename: dataset.filename,
        content: dataset.content,
        content_encoding: dataset.content_encoding || "text",
      }),
    });
    state.klineJob = null;
    renderKlineJob();
  } else if (source === "imported_assets") {
    state.datasetManifest = await fetchJson("/guided/dataset-onboarding/imported-assets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...ctx,
        task_type: state.taskIntake.task_type,
      }),
    });
    state.klineJob = null;
    renderKlineJob();
  } else {
    throw new Error("Use the dedicated online K-line job button for this data source.");
  }
  renderDatasetManifest();
  setWorkspace("full-pipeline");
}

async function startOnlineKlineDownload() {
  if (!state.taskIntake?.task_type) {
    throw new Error("Analyze the task first so the system knows what this dataset is for.");
  }
  const ctx = captureEntryContext();
  const payload = await fetchJson("/guided/dataset-onboarding/online-kline/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...ctx,
      task_type: state.taskIntake.task_type,
      stock_codes: $("kline-stock-codes").value.trim(),
      earliest_date: $("kline-earliest-date").value.trim() || "20190101",
      adjust_type: $("kline-adjust-type").value,
      full_refresh: $("kline-full-refresh").checked,
      update_indexes: $("kline-update-indexes").checked,
    }),
  });
  state.klineJob = payload;
  state.datasetManifest = null;
  renderKlineJob();
  renderDatasetManifest();
  setWorkspace("full-pipeline");
  stopKlineJobPolling();
  state.klineJobPoller = setInterval(async () => {
    try {
      const job = await fetchJson(
        `/guided/dataset-onboarding/online-kline/job?profile=${encodeURIComponent(ctx.profile)}&project_id=${encodeURIComponent(ctx.project_id)}&dataset_id=${encodeURIComponent(ctx.dataset_id)}&run_id=${encodeURIComponent(ctx.run_id)}&job_id=${encodeURIComponent(payload.job_id)}`,
      );
      state.klineJob = job;
      if (job.manifest_preview) {
        state.datasetManifest = job.manifest_preview;
        renderDatasetManifest();
      }
      renderKlineJob();
      if (["completed", "failed"].includes(job.status)) {
        stopKlineJobPolling();
        await loadProjectView(ctx);
      }
    } catch (error) {
      stopKlineJobPolling();
      $("kline-job-logs").textContent = `Job polling stopped: ${error.message}`;
    }
  }, 2500);
}

async function generateWizardArtifacts() {
  if (!state.taskIntake) {
    throw new Error("Analyze the task first.");
  }
  const ctx = captureEntryContext();
  state.wizardBundle = await fetchJson("/guided/run-wizard", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...ctx,
      api_model: $("wizard-api-model").value.trim() || "gpt-oss-20b",
      task_intake: state.taskIntake,
      dataset_manifest: state.datasetManifest,
    }),
  });
  state.activeRun = {
    project_id: state.wizardBundle.run_spec.project_id,
    dataset_id: state.wizardBundle.run_spec.dataset_id,
    run_id: state.wizardBundle.run_spec.run_id,
  };
  applyContextToForms(state.wizardBundle.run_spec);
  await loadProjectView(captureEntryContext());
  await loadRunSpecs();
  await loadRunScopedViews();
  renderWizard();
  setWorkspace("full-pipeline");
}

async function launchWizardPipeline() {
  if (!state.wizardBundle?.launchable) {
    throw new Error("This wizard output is not launchable yet.");
  }
  const runSpec = state.wizardBundle.run_spec || {};
  await fetchJson("/pipeline/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile: runSpec.profile_id,
      mode: runSpec.mode,
      research_goal: runSpec.research_goal,
      run_label: runSpec.run_id,
      project_id: runSpec.project_id,
      dataset_id: runSpec.dataset_id,
      allow_imported_fallback: runSpec.allow_imported_fallback,
      allow_demo_fallback: runSpec.allow_demo_fallback,
      lesson_alias: runSpec.lesson_alias || "alignment_seed0005",
      api_model: runSpec.api_model || "gpt-oss-20b",
      data_dir: runSpec.data_dir || "",
      global_env: runSpec.global_env || {},
      check: false,
      allow_manual_steps: true,
    }),
  });
  state.activeRun = {
    project_id: runSpec.project_id,
    dataset_id: runSpec.dataset_id,
    run_id: runSpec.run_id,
  };
  await loadRunSpecs();
  await loadRunScopedViews();
  setWorkspace("full-pipeline");
}

async function generateAdvancedRunSpec() {
  const ctx = captureProjectForm();
  const body = {
    profile: ctx.profile,
    mode: $("runspec-mode").value,
    research_goal: $("runspec-goal").value.trim(),
    project_id: ctx.project_id,
    dataset_id: ctx.dataset_id,
    run_id: ctx.run_id,
    allow_imported_fallback: ctx.allow_imported_fallback,
    allow_demo_fallback: ctx.allow_demo_fallback,
    api_model: $("runspec-api-model").value.trim() || "gpt-oss-20b",
  };
  state.latestRunSpec = await fetchJson("/console/run-spec", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.activeRun = {
    project_id: state.latestRunSpec.run_spec.project_id,
    dataset_id: state.latestRunSpec.run_spec.dataset_id,
    run_id: state.latestRunSpec.run_spec.run_id,
  };
  applyContextToForms(state.latestRunSpec.run_spec);
  renderRunSpecOutput();
  renderStatusRibbon();
  await loadRunSpecs();
}

async function launchAdvancedPipeline() {
  const ctx = captureProjectForm();
  const pipelineInputs = state.datasetManifest?.pipeline_inputs || {};
  const body = {
    profile: ctx.profile,
    mode: $("runspec-mode").value,
    research_goal: $("runspec-goal").value.trim(),
    run_label: ctx.run_id,
    project_id: ctx.project_id,
    dataset_id: ctx.dataset_id,
    allow_imported_fallback: ctx.allow_imported_fallback,
    allow_demo_fallback: ctx.allow_demo_fallback,
    api_model: $("runspec-api-model").value.trim() || "gpt-oss-20b",
    data_dir: pipelineInputs.data_dir || "",
    global_env: pipelineInputs.global_env || {},
    check: false,
    allow_manual_steps: true,
  };
  await fetchJson("/pipeline/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.activeRun = {
    project_id: ctx.project_id,
    dataset_id: ctx.dataset_id,
    run_id: ctx.run_id,
  };
  await loadRunSpecs();
  await loadRunScopedViews();
  setWorkspace("full-pipeline");
}

async function parseScoringExternalInput(expectBatch) {
  const file = $("scoring-batch-file").files?.[0];
  let raw = $("scoring-signal-json").value.trim();
  if (file) raw = await file.text();
  if (!raw) {
    throw new Error(expectBatch ? "Provide a JSON array of signal records or upload a JSON file." : "Provide a signal JSON object.");
  }
  const parsed = JSON.parse(raw);
  if (expectBatch) {
    if (Array.isArray(parsed)) return parsed;
    return [parsed];
  }
  if (Array.isArray(parsed)) {
    throw new Error("Single-signal mode expects one JSON object, not an array.");
  }
  return parsed;
}

async function runScoring() {
  const sourceMode = $("scoring-source").value;
  const inputMode = $("scoring-input-mode").value;
  const recordedRun = $("scoring-recorded-run").value;
  if (sourceMode === "current_workflow" && !state.lessonSet?.final_lesson_state_json) {
    throw new Error("The active run does not have a current workflow final lesson set yet.");
  }
  const common = {
    profile: state.profile,
    lesson_alias: sourceMode === "imported" ? $("scoring-lesson-alias").value : "",
    final_lesson_state_json: sourceMode === "current_workflow" ? state.lessonSet?.final_lesson_state_json || "" : "",
    prompt_only: $("scoring-prompt-only").checked,
    persist_run: false,
    schema_from_run: recordedRun,
  };

  if (inputMode === "recorded_replay") {
    const body = {
      ...common,
      from_recorded_run: recordedRun,
      signal_date: $("scoring-signal-date").value.trim(),
      symbol: $("scoring-symbol").value.trim(),
    };
    const payload = await fetchJson("/score/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderScoringSingle(payload);
    return;
  }

  if (inputMode === "external_single") {
    const signalRecord = await parseScoringExternalInput(false);
    const payload = await fetchJson("/score/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...common,
        signal_record: signalRecord,
      }),
    });
    renderScoringSingle(payload);
    return;
  }

  const signalRecords = await parseScoringExternalInput(true);
  const payload = await fetchJson("/score/live-batch-external", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...common,
      signal_records: signalRecords,
    }),
  });
  renderScoringBatch(payload);
}

async function sendSimpleChatMessage() {
  const input = $("simple-chat-input");
  const message = input.value.trim();
  if (!message) return;
  const ctx = captureEntryContext();
  const onlineKlineEnabled = Boolean($("simple-natural-kline-toggle")?.checked);
  const stockCodes = Array.from(new Set(message.match(/\d{6}/g) || []));
  if (onlineKlineEnabled && stockCodes.length && $("simple-kline-codes") && !$("simple-kline-codes").value.trim()) {
    $("simple-kline-codes").value = stockCodes.join(", ");
  }
  setSimpleKlineDefaults();
  state.chatMessages.push({ role: "user", content: message });
  input.value = "";
  state.simpleTyping = true;
  renderSimpleChatThread();

  try {
    const payload = await fetchJson("/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...ctx,
        session_id: state.chatSession?.session_id || "",
        mode: "simple",
        message,
        attachments: onlineKlineEnabled
          ? [{ kind: "ui_preference", online_kline_enabled: true, note: "User allows Simple Mode to fetch recent K-line data when the visible request clearly asks to inspect a stock." }]
          : [],
      }),
    });
    await applyChatPayload(payload);
    state.chatMessages.push({
      role: "assistant",
      content: payload.assistant_message_zh || "我已经更新了 task_state。",
    });
    if (payload.guided_task_intake) {
      state.taskIntake = payload.guided_task_intake;
      $("intake-user-request").value = message;
      renderTaskIntake();
    }
    renderSimpleMode();
    renderStatusRibbon();
  } catch (error) {
    state.chatMessages.push({
      role: "assistant",
      content: `请求失败：${error.message}`,
    });
    renderSimpleChatThread();
  } finally {
    state.simpleTyping = false;
    renderSimpleChatThread();
  }
}

async function sendSimpleIntent(message) {
  if (!message) return;
  $("simple-chat-input").value = message;
  await sendSimpleChatMessage();
}

async function executeExpertChatAction(actionId, options = {}) {
  const ctx = captureEntryContext();
  const payload = await fetchJson("/chat/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...ctx,
      session_id: state.chatSession?.session_id || state.taskState?.session_id || "",
      action_id: actionId,
      confirm: Boolean(options.confirm),
      task_state: state.taskState || {},
      file_payload: options.file_payload || null,
      kline_params: options.kline_params || {},
      signal_record: options.signal_record || null,
      scoring_payload: options.scoring_payload || {},
      api_model: $("wizard-api-model")?.value?.trim() || "gpt-oss-20b",
    }),
  });
  await applyChatPayload(payload);
  state.expertChatMessages.push({
    role: "assistant",
    content: payload.assistant_message_zh || `动作 ${actionId} 已执行。`,
  });
  renderExpertWorkbench();
  renderStatusRibbon();
  return payload;
}

async function sendExpertChatMessage() {
  const input = $("expert-chat-input");
  const fileInput = $("expert-chat-upload-file");
  const file = fileInput?.files?.[0] || null;
  let message = input.value.trim();
  if (!message && file) {
    message = "我上传了一份市场数据，请帮我检查能不能运行完整研究流程，并给出下一步研究配置建议。";
  }
  if (!message) return;
  const ctx = captureEntryContext();
  state.expertChatMessages.push({ role: "user", content: file ? `${message}\n\n附件：${file.name}` : message });
  input.value = "";
  state.expertTyping = true;
  renderExpertChatThread();
  try {
    const payload = await fetchJson("/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...ctx,
        session_id: state.chatSession?.session_id || "",
        mode: "expert",
        message,
        attachments: file ? [{ kind: "uploaded_file_pending_validation", filename: file.name }] : [],
      }),
    });
    await applyChatPayload(payload);
    state.expertChatMessages.push({
      role: "assistant",
      content: payload.assistant_message_zh || "我已经更新了研究任务状态。",
    });
    if (payload.guided_task_intake) {
      state.taskIntake = payload.guided_task_intake;
      $("intake-user-request").value = message;
      renderTaskIntake();
    }
    if (file) {
      const filePayload = await readFilePayload(file);
      await executeExpertChatAction("upload_dataset", { file_payload: filePayload });
      if ($("expert-chat-upload-name")) $("expert-chat-upload-name").textContent = `已上传并完成检查：${file.name}`;
      if (fileInput) fileInput.value = "";
      state.expertAttachedFilename = "";
    }
    renderExpertWorkbench();
    renderDatasetManifest();
    renderStatusRibbon();
  } catch (error) {
    state.expertChatMessages.push({
      role: "assistant",
      content: `专业模式暂时不能继续：${error.message}`,
    });
  } finally {
    state.expertTyping = false;
    renderExpertWorkbench();
  }
}

async function applyChatPayload(payload) {
  state.chatSession = payload.session || state.chatSession || null;
  state.taskState = payload.task_state || state.taskState || null;
  state.chatRecommendedActions = payload.recommended_actions || state.chatRecommendedActions || [];
  state.chatArtifactLinks = payload.artifact_links || state.chatArtifactLinks || [];
  if (payload.dataset_manifest) {
    state.datasetManifest = payload.dataset_manifest;
    renderDatasetManifest();
  }
  if (payload.wizard_bundle) {
    state.wizardBundle = payload.wizard_bundle;
    renderWizard();
    await loadRunSpecs();
  }
  if (payload.workflow_result) {
    await loadRunSpecs();
  }
  if (payload.kline_job) {
    state.klineJob = payload.kline_job;
    renderKlineJob();
    startSimpleKlinePolling(payload.kline_job);
  }
  if (payload.scoring_result) {
    state.simpleScoringResult = payload.scoring_result;
    renderSimpleScoringResult();
  }
  if (state.taskState) {
    state.activeRun = {
      project_id: state.taskState.project_id,
      dataset_id: state.taskState.dataset_id,
      run_id: state.taskState.run_id,
    };
    applyContextToForms({
      ...captureEntryContext(),
      project_id: state.taskState.project_id,
      dataset_id: state.taskState.dataset_id,
      run_id: state.taskState.run_id,
      profile_id: state.taskState.profile || state.profile,
      allow_imported_fallback: state.taskState.allow_imported_fallback,
      allow_demo_fallback: state.taskState.allow_demo_fallback,
    });
  }
  await loadProjectView(captureEntryContext());
  const taskType = state.taskState?.task_type || "";
  const shouldShowWorkflowStatus = Boolean(payload.workflow_result)
    && Boolean(state.taskState?.artifact_exists?.run_spec_json)
    && !payload.scoring_result
    && taskType !== "scoring_only";
  if (shouldShowWorkflowStatus) {
    await loadRunScopedViews();
    try {
      const status = await refreshSimpleRunStatus({ appendMessage: false });
      if (status && ["running", "pending", "queued"].includes(status.workflow_status)) {
        startSimpleWorkflowPolling({ appendMessage: false });
      }
    } catch (error) {
      // A run may have a spec before any workflow result exists. Keep Simple Mode usable.
    }
  }
}

async function executeChatAction(actionId, options = {}) {
  const ctx = captureEntryContext();
  const payload = await fetchJson("/chat/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...ctx,
      session_id: state.chatSession?.session_id || state.taskState?.session_id || "",
      action_id: actionId,
      confirm: Boolean(options.confirm),
      task_state: state.taskState || {},
      file_payload: options.file_payload || null,
      kline_params: options.kline_params || {},
      signal_record: options.signal_record || null,
      scoring_payload: options.scoring_payload || {},
      api_model: $("wizard-api-model")?.value?.trim() || "gpt-oss-20b",
    }),
  });
  await applyChatPayload(payload);
  state.chatMessages.push({
    role: "assistant",
    content: payload.assistant_message_zh || `动作 ${actionId} 已执行。`,
  });
  renderSimpleMode();
  renderStatusRibbon();
  return payload;
}

function stopSimpleKlinePolling() {
  if (state.simpleKlinePoller) {
    clearInterval(state.simpleKlinePoller);
    state.simpleKlinePoller = null;
  }
}

function startSimpleKlinePolling(job) {
  if (!job?.job_id) return;
  stopSimpleKlinePolling();
  const ctx = captureEntryContext();
  state.simpleKlinePoller = setInterval(async () => {
    try {
      const payload = await fetchJson(
        `/guided/dataset-onboarding/online-kline/job?profile=${encodeURIComponent(ctx.profile)}&project_id=${encodeURIComponent(ctx.project_id)}&dataset_id=${encodeURIComponent(ctx.dataset_id)}&run_id=${encodeURIComponent(ctx.run_id)}&job_id=${encodeURIComponent(job.job_id)}`,
      );
      state.klineJob = payload;
      renderKlineJob();
      if (payload.manifest_preview) {
        state.datasetManifest = payload.manifest_preview;
        renderDatasetManifest();
      }
      if (["completed", "failed"].includes(payload.status)) {
        stopSimpleKlinePolling();
        if (payload.manifest_preview) {
          await executeChatAction("sync_dataset_manifest", { confirm: true });
        } else {
          state.chatMessages.push({ role: "assistant", content: `在线 K 线任务结束：${payload.status}。未发现可同步的 manifest。` });
          renderSimpleChatThread();
        }
      }
    } catch (error) {
      stopSimpleKlinePolling();
      state.chatMessages.push({ role: "assistant", content: `K 线任务轮询停止：${error.message}` });
      renderSimpleChatThread();
    }
  }, 3000);
}

function wireForms() {
  setSimpleKlineDefaults();
  $("mode-simple-toggle").addEventListener("click", () => setWorkspace("simple"));
  $("mode-expert-toggle").addEventListener("click", () => setWorkspace("full-pipeline"));
  if ($("expert-chat-form")) {
    $("expert-chat-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendExpertChatMessage();
    });
  }
  if ($("expert-chat-upload-file")) {
    $("expert-chat-upload-file").addEventListener("change", () => {
      const file = $("expert-chat-upload-file").files?.[0];
      state.expertAttachedFilename = file?.name || "";
      if ($("expert-chat-upload-name")) {
        $("expert-chat-upload-name").textContent = file ? `已选择：${file.name}` : "未选择文件";
      }
    });
  }
  $("simple-chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendSimpleChatMessage();
  });
  document.querySelectorAll("[data-simple-intent]").forEach((button) => {
    button.addEventListener("click", async () => {
      await sendSimpleIntent(button.dataset.simpleIntent || "");
    });
  });
  if ($("simple-chat-upload-file")) {
    $("simple-chat-upload-file").addEventListener("change", () => {
      const file = $("simple-chat-upload-file").files?.[0];
      state.simpleAttachedFilename = file?.name || "";
      if ($("simple-chat-upload-name")) {
        $("simple-chat-upload-name").textContent = file ? `已选择，尚未上传：${file.name}` : "尚未选择文件";
      }
      if ($("simple-chat-upload-submit")) {
        $("simple-chat-upload-submit").classList.toggle("is-hidden", !file);
      }
    });
  }
  if ($("simple-upload-file")) {
    $("simple-upload-file").addEventListener("change", () => {
      const file = $("simple-upload-file").files?.[0];
      state.simpleAttachedFilename = file?.name || state.simpleAttachedFilename;
      if ($("simple-chat-upload-name") && file) {
        $("simple-chat-upload-name").textContent = `已选择，尚未上传：${file.name}`;
      }
      if ($("simple-upload-status")) {
        $("simple-upload-status").textContent = file ? `已选择，尚未上传：${file.name}` : "尚未选择文件。";
      }
      if ($("simple-chat-upload-submit")) {
        $("simple-chat-upload-submit").classList.toggle("is-hidden", !file);
      }
    });
  }
  if ($("simple-chat-upload-submit")) {
    $("simple-chat-upload-submit").addEventListener("click", async () => {
      await handleSimpleAction("upload_dataset");
    });
  }
  $("simple-open-expert").addEventListener("click", () => setWorkspace("full-pipeline"));
  if ($("simple-upload-button")) {
    $("simple-upload-button").addEventListener("click", async () => {
      await handleSimpleAction("upload_dataset");
    });
  }
  if ($("simple-kline-button")) {
    $("simple-kline-button").addEventListener("click", async () => {
      await handleSimpleAction("start_online_kline_download");
    });
  }
  if ($("simple-scoring-mode")) {
    $("simple-scoring-mode").addEventListener("change", () => {
      renderSimpleScoringModeControls();
      renderSimpleActions();
    });
  }
  if ($("simple-scoring-button")) {
    $("simple-scoring-button").addEventListener("click", async () => {
      try {
        state.simpleTyping = true;
        renderSimpleChatThread();
        await runSimpleScoringFlow();
      } catch (error) {
        state.chatMessages.push({
          role: "assistant",
          content: `评分流程暂时不能继续：${error.message}\n如果你是在看单只 A 股，请直接说“今天帮我看看 600519”；如果你要评分自定义候选信号，请上传或粘贴结构化 signal record。`,
        });
        renderSimpleChatThread();
        renderSimpleScoringResult({
          mode: $("simple-scoring-mode")?.value || "prompt_only",
          result_type: "client_side_input_error",
          scoring_provenance: {
            model_called: false,
            result_valid_for_research: false,
            lesson_source: "-",
            teacher_source: "-",
            fallback_used: false,
          },
          signal_input_manifest: {
            valid: false,
            record_count: 0,
            missing_columns: ["structured_signal_record"],
          },
          summary_zh: error.message,
        });
      } finally {
        state.simpleTyping = false;
        renderSimpleChatThread();
      }
    });
  }
  if ($("simple-scoring-expert-button")) {
    $("simple-scoring-expert-button").addEventListener("click", () => setWorkspace("scoring"));
  }

  $("dashboard-go-full-pipeline").addEventListener("click", () => setWorkspace("full-pipeline"));
  $("dashboard-go-scoring").addEventListener("click", () => setWorkspace("scoring"));
  $("dashboard-go-library").addEventListener("click", () => setWorkspace("library"));
  $("dashboard-go-provenance").addEventListener("click", () => setWorkspace("provenance"));

  $("dataset-source-type").addEventListener("change", () => {
    renderDatasetSourcePanels();
  });
  $("scoring-input-mode").addEventListener("change", () => renderScoringControls());
  $("scoring-source").addEventListener("change", () => renderScoringControls());
  $("library-lesson-alias").addEventListener("change", async (event) => {
    await loadImportedLessonDetail(event.target.value);
  });
  if ($("lesson-lab-imported-alias")) {
    $("lesson-lab-imported-alias").addEventListener("change", async (event) => {
      await loadImportedLessonDetail(event.target.value);
    });
  }
  $("intake-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await analyzeTaskIntake();
    } catch (error) {
      $("intake-summary-text").textContent = error.message;
    }
  });
  $("intake-go-dataset").addEventListener("click", () => setWorkspace("full-pipeline"));

  $("dataset-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await validateDatasetManifest();
    } catch (error) {
      $("dataset-issues").innerHTML = `<div class="notice-card compact warn-card"><strong>Dataset onboarding error</strong><p>${escapeHtml(error.message)}</p></div>`;
      setWorkspace("full-pipeline");
    }
  });
  $("dataset-go-wizard").addEventListener("click", () => setWorkspace("full-pipeline"));
  $("dataset-imported-assets-button").addEventListener("click", async () => {
    try {
      await validateDatasetManifest();
    } catch (error) {
      $("dataset-issues").innerHTML = `<div class="notice-card compact warn-card"><strong>Dataset onboarding error</strong><p>${escapeHtml(error.message)}</p></div>`;
      setWorkspace("full-pipeline");
    }
  });
  $("kline-start-job").addEventListener("click", async () => {
    try {
      await startOnlineKlineDownload();
    } catch (error) {
      $("kline-job-logs").textContent = error.message;
      setWorkspace("full-pipeline");
    }
  });

  $("wizard-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await generateWizardArtifacts();
    } catch (error) {
      $("wizard-summary-text").textContent = error.message;
      setWorkspace("full-pipeline");
    }
  });
  $("wizard-launch").addEventListener("click", async () => {
    try {
      await launchWizardPipeline();
    } catch (error) {
      $("wizard-summary-text").textContent = error.message;
      setWorkspace("full-pipeline");
    }
  });
  $("wizard-open-advanced").addEventListener("click", () => setWorkspace("advanced"));

  $("project-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ctx = captureProjectForm();
    applyContextToForms(ctx);
    await loadProjectView(ctx);
    await loadRunSpecs();
    setWorkspace("provenance");
  });

  $("runspec-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await generateAdvancedRunSpec();
  });
  $("launch-pipeline-button").addEventListener("click", async () => {
    await launchAdvancedPipeline();
  });

  $("scoring-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await runScoring();
      setWorkspace("scoring");
    } catch (error) {
      $("scoring-raw").textContent = error.message;
      $("scoring-total-score").textContent = "Error";
      $("scoring-short-reason").textContent = error.message;
      $("scoring-batch-body").innerHTML = "";
      setWorkspace("scoring");
    }
  });
}

async function bootstrap() {
  wireWorkspaceTabs();
  wireForms();
  renderDatasetSourcePanels();
  renderScoringControls();
  applyContextToForms({
    project_id: "default-project",
    dataset_id: "default-dataset",
    run_id: "guided-run",
    profile_id: state.profile,
    allow_imported_fallback: true,
    allow_demo_fallback: false,
  });
  await loadProjectView(captureEntryContext());
  await refreshBaseData();
  await loadRunSpecs();
  renderTaskIntake();
  renderDatasetManifest();
  renderKlineJob();
  renderWizard();
  renderSimpleMode();
  setWorkspace("simple");
  if (state.runSpecs.length) {
    const last = state.runSpecs[state.runSpecs.length - 1];
    state.activeRun = {
      project_id: last.project_id,
      dataset_id: last.dataset_id,
      run_id: last.run_id,
    };
    applyContextToForms({
      ...captureEntryContext(),
      project_id: last.project_id,
      dataset_id: last.dataset_id,
      run_id: last.run_id,
    });
    await loadProjectView(captureEntryContext());
    await loadRunScopedViews();
    try {
      await refreshSimpleRunStatus({ appendMessage: false });
    } catch (error) {
      // Existing spec-only runs may not have complete workflow artifacts yet.
    }
  }
  renderStatusRibbon();
}

bootstrap().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<pre style="padding:24px;color:#b44a2d;">Bootstrap failed:\n${escapeHtml(error.message)}</pre>`;
});
