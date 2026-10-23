import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import ConnectPage from "./ConnectPage";
import PhoneCapturePage from "./PhoneCapturePage";
import { bootstrapOperatorAuth } from "./api";
import "./styles.css";
import "./application.css";

bootstrapOperatorAuth();

const pathname = window.location.pathname;
const screen =
  pathname === "/connect" ? (
    <ConnectPage />
  ) : pathname === "/phone" ? (
    <PhoneCapturePage />
  ) : (
    <App />
  );
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>{screen}</React.StrictMode>,
);
