import { afterEach, expect, it, vi } from "vitest";
import { markUiFrame, uiRate } from "./clientMetrics";

afterEach(() => vi.restoreAllMocks());
it("counts displayed inference images once despite duplicate or older loads", () => {
  const clock = vi.spyOn(performance, "now").mockReturnValue(1000);
  markUiFrame("unique-images", "session", 1);
  clock.mockReturnValue(2000);
  markUiFrame("unique-images", "session", 1);
  markUiFrame("unique-images", "session", 0);
  markUiFrame("unique-images", "session", 2);
  expect(uiRate("unique-images")).toMatchObject({
    fps: 2,
    total: 2,
    window_count: 2,
  });
  clock.mockReturnValue(62001);
  expect(uiRate("unique-images")).toMatchObject({
    fps: 0,
    total: 2,
    window_count: 0,
  });
});
it("does not invent UI FPS without a measured interval or reuse an earlier session", () => {
  const clock = vi.spyOn(performance, "now").mockReturnValue(1000);
  expect(uiRate("never-displayed")).toBeNull();
  markUiFrame("restart-images", "old", 50);
  expect(uiRate("restart-images")?.fps).toBeNull();
  clock.mockReturnValue(2000);
  markUiFrame("restart-images", "new", 1);
  expect(uiRate("restart-images")).toMatchObject({
    fps: null,
    total: 1,
    window_count: 1,
  });
});
