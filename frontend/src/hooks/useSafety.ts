import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, errorText } from "../api";
import type { Camera, Incident, Info } from "../types";
import { commandTransition, initialCommand } from "../machines/command";
export function useSafety() {
  const [info, setInfo] = useState<Info | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [recent, setRecent] = useState<Incident[]>([]);
  const [activeIncidents, setActiveIncidents] = useState<Incident[]>([]);
  const [feedback, setFeedback] = useState("unreviewed");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [journalError, setJournalError] = useState("");
  const [total, setTotal] = useState(0);
  const [filter, setFilter] = useState("");
  const [incidentType, setIncidentType] = useState("");
  const [incidentCamera, setIncidentCamera] = useState("");
  const [offset, setOffset] = useState(0);
  const [connection, setConnection] = useState<
    "connecting" | "online" | "offline"
  >("connecting");
  const [accessDenied, setAccessDenied] = useState(false);
  const [command, setCommand] = useState(initialCommand);
  const executing = useRef(false);
  const live = useRef(false);
  const requestId = useRef(0);
  const incidentsRequestId = useRef(0);
  const refreshState = useCallback(async () => {
    const id = ++requestId.current;
    const [nextInfo, nextCameras] = await Promise.all([
      api<Info>("/system/info"),
      api<Camera[]>("/cameras"),
    ]);
    if (id !== requestId.current) return;
    live.current = nextCameras.some((c) =>
      ["starting", "running", "stopping", "uploading"].includes(c.status),
    );
    setInfo(nextInfo);
    setCameras(nextCameras);
    setAccessDenied(false);
    setConnection("online");
  }, []);
  const incidentParams = useCallback(() => {
    const params = new URLSearchParams();
    if (filter) params.set("status", filter);
    if (incidentType) params.append("type", incidentType);
    if (incidentCamera) params.append("camera_id", incidentCamera);
    if (feedback) params.set("feedback", feedback);
    if (fromDate)
      params.set("from_time", new Date(fromDate + "T00:00:00").toISOString());
    if (toDate)
      params.set("to_time", new Date(toDate + "T23:59:59.999").toISOString());
    return params;
  }, [filter, incidentType, incidentCamera, feedback, fromDate, toDate]);
  const refreshIncidents = useCallback(async () => {
    const id = ++incidentsRequestId.current;
    const params = incidentParams();
    params.set("limit", "30");
    params.set("offset", String(offset));
    const list = await api<{ items: Incident[]; total: number }>(
      `/incidents?${params.toString()}`,
    );
    if (id !== incidentsRequestId.current) return;
    // In-place update: pagination and scroll position survive the poll.
    setIncidents(list.items);
    setTotal(list.total);
    setIncidentLoading(false);
    setJournalError("");
  }, [incidentParams, offset]);
  const refreshFeed = useCallback(async () => {
    const [recent, active] = await Promise.all([
      api<{ items: Incident[] }>("/incidents?limit=8"),
      api<{ items: Incident[] }>("/incidents?limit=100&status=active"),
    ]);
    setRecent(recent.items);
    setActiveIncidents(active.items);
  }, []);
  const refresh = useCallback(async () => {
    await refreshState();
    await Promise.all([refreshIncidents(), refreshFeed()]);
  }, [refreshState, refreshIncidents, refreshFeed]);
  useEffect(() => {
    let cancelled = false;
    let authFailed = false;
    let fast: ReturnType<typeof setTimeout>;
    let slow: ReturnType<typeof setTimeout>;
    // Live state polls up to 5 Hz and slows down when idle. This is not analytics FPS;
    // incident history is heavier and polls every 5 seconds instead,
    // so the list no longer churns under the operator's hands.
    const pollFast = async () => {
      try {
        if (!cancelled) await refreshState();
      } catch (e) {
        if (!cancelled) {
          authFailed = e instanceof ApiError && e.status === 401;
          setAccessDenied(authFailed);
          setConnection("offline");
        }
      } finally {
        if (!cancelled && !authFailed)
          fast = setTimeout(pollFast, live.current ? 200 : 1500);
      }
    };
    const pollSlow = async () => {
      try {
        if (!cancelled) await refreshIncidents();
      } catch (e) {
        if (!cancelled) {
          setJournalError(errorText(e));
          setIncidentLoading(false);
        }
      } finally {
        if (!cancelled) slow = setTimeout(pollSlow, 5000);
      }
    };
    setIncidentLoading(true);
    void pollFast();
    void pollSlow();
    let feed: ReturnType<typeof setTimeout>;
    const pollFeed = async () => {
      try {
        if (!cancelled) await refreshFeed();
      } catch {
        /* live state owns connectivity */
      } finally {
        if (!cancelled) feed = setTimeout(pollFeed, 2000);
      }
    };
    void pollFeed();
    return () => {
      cancelled = true;
      requestId.current++;
      incidentsRequestId.current++;
      clearTimeout(fast);
      clearTimeout(slow);
      clearTimeout(feed);
    };
  }, [refreshState, refreshIncidents, refreshFeed]);
  async function act(work: () => Promise<void>) {
    if (executing.current) return false;
    executing.current = true;
    setCommand((old) => commandTransition(old, { type: "BEGIN" }));
    try {
      await work();
      setCommand((old) => commandTransition(old, { type: "DONE" }));
      await refresh().catch(() => setConnection("offline"));
      return true;
    } catch (e) {
      setCommand((old) =>
        commandTransition(old, { type: "FAIL", message: errorText(e) }),
      );
      return false;
    } finally {
      executing.current = false;
    }
  }
  return {
    info,
    recent,
    activeIncidents,
    feedback,
    fromDate,
    toDate,
    incidentLoading,
    journalError,
    incidentParams,
    setFeedback: (value: string) => {
      setFeedback(value);
      setOffset(0);
    },
    setFromDate: (value: string) => {
      setFromDate(value);
      setOffset(0);
    },
    setToDate: (value: string) => {
      setToDate(value);
      setOffset(0);
    },
    cameras,
    incidents,
    total,
    filter,
    incidentType,
    incidentCamera,
    offset,
    setFilter: (value: string) => {
      setFilter(value);
      setOffset(0);
    },
    setIncidentType: (value: string) => {
      setIncidentType(value);
      setOffset(0);
    },
    setIncidentCamera: (value: string) => {
      setIncidentCamera(value);
      setOffset(0);
    },
    setOffset,
    connection,
    accessDenied,
    command,
    busy: command.phase === "executing",
    act,
    refresh,
    clearError: () =>
      setCommand((old) =>
        old.phase === "failed"
          ? commandTransition(old, { type: "DISMISS" })
          : old,
      ),
  };
}
