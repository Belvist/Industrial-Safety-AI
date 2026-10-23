import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, errorText } from "../api";
import type { Camera, Incident, Info } from "../types";
import { commandTransition, initialCommand } from "../machines/command";
export function useSafety() {
  const [info, setInfo] = useState<Info | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
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
  const requestId = useRef(0);
  const refreshState = useCallback(async () => {
    const id = ++requestId.current;
    const [nextInfo, nextCameras] = await Promise.all([
      api<Info>("/system/info"),
      api<Camera[]>("/cameras"),
    ]);
    if (id !== requestId.current) return;
    setInfo(nextInfo);
    setCameras(nextCameras);
    setAccessDenied(false);
    setConnection("online");
  }, []);
  const refreshIncidents = useCallback(async () => {
    const id = ++requestId.current;
    const params = new URLSearchParams({
      limit: "30",
      offset: String(offset),
    });
    if (filter) params.set("status", filter);
    if (incidentType) params.append("type", incidentType);
    if (incidentCamera) params.append("camera_id", incidentCamera);
    const list = await api<{ items: Incident[]; total: number }>(
      `/incidents?${params.toString()}`,
    );
    if (id !== requestId.current) return;
    // In-place update: pagination and scroll position survive the poll.
    setIncidents(list.items);
    setTotal(list.total);
  }, [filter, incidentType, incidentCamera, offset]);
  const refresh = useCallback(async () => {
    await refreshState();
    await refreshIncidents();
  }, [refreshState, refreshIncidents]);
  useEffect(() => {
    let cancelled = false;
    let authFailed = false;
    let fast: ReturnType<typeof setTimeout>;
    let slow: ReturnType<typeof setTimeout>;
    // Live state (info + cameras + frame) polls every second;
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
        if (!cancelled && !authFailed) fast = setTimeout(pollFast, 1000);
      }
    };
    const pollSlow = async () => {
      try {
        if (!cancelled) await refreshIncidents();
      } catch {
        // incidents refresh is best-effort; state poll owns the connection flag
      } finally {
        if (!cancelled) slow = setTimeout(pollSlow, 5000);
      }
    };
    void pollFast();
    void pollSlow();
    return () => {
      cancelled = true;
      requestId.current++;
      clearTimeout(fast);
      clearTimeout(slow);
    };
  }, [refreshState, refreshIncidents]);
  async function act(work: () => Promise<void>) {
    if (executing.current) return;
    executing.current = true;
    setCommand((old) => commandTransition(old, { type: "BEGIN" }));
    try {
      await work();
      setCommand((old) => commandTransition(old, { type: "DONE" }));
      await refresh().catch(() => setConnection("offline"));
    } catch (e) {
      setCommand((old) =>
        commandTransition(old, { type: "FAIL", message: errorText(e) }),
      );
    } finally {
      executing.current = false;
    }
  }
  return {
    info,
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
