import "../app.css";
import { useState } from "react";
import IsaacStreamViewer from "../components/IsaacStreamViewer";
import { GearIcon } from "@radix-ui/react-icons";
import SidebarMenu from "../components/SidebarMenu";

export default function SimulationPage() {
  const [openMenu, setOpenMenu] = useState(null);
  const [selectedRobot, setSelectedRobot] = useState(null);
  const [selectedGripper, setSelectedGripper] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [connectionError, setConnectionError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleStatusChange = (status, errMsg = null) => {
    setConnectionStatus(status);
    setConnectionError(errMsg);
  };

  const handleCaptureFrame = async () => {
    try {
      console.log("🎯 Running trajectory...");
      const response = await fetch("http://10.10.10.4:49101/run_trajectory", {
        method: "POST",
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Trajectory run failed");
      }

      // Download tactile_data.json
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "tactile_data.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      console.log("✅ Frame captured and file downloaded.");
    } catch (err) {
      console.error("Error running trajectory:", err);
      setConnectionError(err.message);
    } finally {
      setIsRunning(false);
    }
  };

  const statusColor =
    connectionStatus === "connected"
      ? "bg-green-400"
      : connectionStatus === "connecting"
      ? "bg-yellow-400"
      : connectionStatus === "error"
      ? "bg-red-400"
      : "bg-gray-400";

  const statusText =
    connectionStatus === "connected"
      ? "Connected"
      : connectionStatus === "connecting"
      ? "Connecting to Isaac Sim..."
      : connectionStatus === "error"
      ? "Connection Failed"
      : "Disconnected";

  return (
    <div className="h-screen flex flex-col bg-black text-green-400 font-orbitron">
      {/* HEADER */}
      <header className="bg-black/90 border-b border-green-500/30 p-4 shadow-md flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold neon-text">Fluxa</h1>
        </div>
        <button className="neon-button flex items-center space-x-2 px-3 py-2 rounded-lg">
          <GearIcon />
          <span>Settings</span>
        </button>
      </header>

      {/* MAIN BODY */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT CONTROL PANEL */}
        <aside className="w-80 bg-black/80 border-r border-green-500/20 p-4 overflow-y-auto">
          <div className="space-y-6">
            {/* Connection Status */}
            <div className="bg-green-950/30 p-3 rounded-lg border border-green-500/20">
              <h3 className="font-semibold mb-2">Connection Status</h3>
              <div className="flex items-center text-green-400">
                {/* Spinner only when connecting */}
                {connectionStatus === "connecting" && <div className="spinner"></div>}

                {/* Status light */}
                <div className={`status-light ${connectionStatus}`}></div>

                {/* Text */}
                <span className="ml-2">
                  {connectionStatus === "connected" && "Connected"}
                  {connectionStatus === "connecting" && "Connecting to Isaac Sim..."}
                  {connectionStatus === "error" && "Connection Failed"}
                  {connectionStatus === "disconnected" && "Disconnected"}
                </span>
              </div>

              {/* Error message */}
              {connectionError && (
                <p className="text-red-400 text-sm mt-2">{connectionError}</p>
              )}
            </div>


            {/* Robot Type */}
            <SidebarMenu
              title="Robot Type"
              options={["Franka", "XArm7", "Custom"]}
              selected={selectedRobot}
              setSelected={setSelectedRobot}
              openMenu={openMenu}
              setOpenMenu={setOpenMenu}
            />

            {/* Gripper Type */}
            <SidebarMenu
              title="Gripper Type"
              options={["UMI", "GelSight", "Custom"]}
              selected={selectedGripper}
              setSelected={setSelectedGripper}
              openMenu={openMenu}
              setOpenMenu={setOpenMenu}
            />

            {/* Object Type */}
            <SidebarMenu
              title="Object Type"
              options={["Blob", "Yarn", "Fabric"]}
              selected={selectedObject}
              setSelected={setSelectedObject}
              openMenu={openMenu}
              setOpenMenu={setOpenMenu}
            />

            {/* Dataset Format */}
            <SidebarMenu
              title="Dataset Format"
              options={["Lerobot", "Pickle", "MCAP", "HDF5"]}
              selected={selectedDataset}
              setSelected={setSelectedDataset}
              openMenu={openMenu}
              setOpenMenu={setOpenMenu}
            />


            {/* Actions */}
            <div className="bg-green-950/30 p-3 rounded-lg border border-green-500/20 space-y-2">
              <h3 className="font-semibold mb-2">Actions</h3>
              <button className="w-full neon-button px-4 py-2 rounded"
                onClick={handleCaptureFrame}>
                Capture Frame
              </button>
              <button className="w-full neon-button px-4 py-2 rounded">
                Generate 100 Frames
              </button>
              <button className="w-full neon-button px-4 py-2 rounded bg-teal-400/30 hover:bg-teal-400/60 transition">
                Clear Scene
              </button>
            </div>

            {/* Controls Info */}
            <div className="bg-green-950/30 p-3 rounded-lg border border-green-500/20 text-xs text-green-300/80">
              <h3 className="font-semibold mb-2">Controls</h3>
              <ul className="space-y-1">
                <li>• Click viewport to focus</li>
                <li>• ALT + Left Mouse: Orbit</li>
                <li>• ALT + Right Mouse: Zoom</li>
                <li>• Middle Mouse: Pan</li>
                <li>• Right Mouse: Fly mode (WASD)</li>
              </ul>
            </div>
          </div>
        </aside>

        {/* VIDEO STREAM */}
        <main className="flex-1 bg-black relative">
          <IsaacStreamViewer 
            onStatusChange={handleStatusChange}
            connectionStatus={connectionStatus}
          />
        </main>
      </div>
    </div>
  );
}
