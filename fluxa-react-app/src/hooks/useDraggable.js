import { useRef, useState, useEffect } from "react";

export default function useDraggable(initialPos = { x: 0, y: 0 }, constraints = {}) {
  const ref = useRef(null);
  const [pos, setPos] = useState(initialPos);
  const [edge, setEdge] = useState("left"); // "left", "right", "top", "bottom"
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Helper function to get panel dimensions and constraints
    const getLayoutData = () => {
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const headerHeight = constraints.minY || 115;
      const panelWidth = el.offsetWidth;
      const panelHeight = el.offsetHeight;
      
      return { windowWidth, windowHeight, headerHeight, panelWidth, panelHeight };
    };

    const onMouseDown = (e) => {
      // 1. Calculate the initial offset of the mouse click inside the panel
      const offset = {
        x: e.clientX - pos.x,
        y: e.clientY - pos.y,
      };

      setDragging(true);

      const onMouseMove = (e) => {
        const { windowWidth, windowHeight, headerHeight, panelWidth, panelHeight } = getLayoutData();

        // 2. Calculate new real-time position
        let newX = e.clientX - offset.x;
        let newY = e.clientY - offset.y;
        
        // 3. Clamp new position within window bounds (optional but recommended)
        newX = Math.max(constraints.minX || 0, Math.min(newX, windowWidth - panelWidth));
        newY = Math.max(headerHeight, Math.min(newY, windowHeight - panelHeight));

        // Update position for smooth dragging
        setPos({ x: newX, y: newY });
        
        // --- EDGE DETECTION (STAYS THE SAME) ---
        // This still detects the closest edge for the SNAP on mouseUp
        const mouseX = e.clientX;
        const mouseY = e.clientY;

        const distToLeft = mouseX;
        const distToRight = windowWidth - mouseX;
        const distToTop = mouseY - headerHeight;
        const distToBottom = windowHeight - mouseY;

        const minDist = Math.min(distToLeft, distToRight, distToTop, distToBottom);

        if (minDist === distToLeft) {
          setEdge("left");
        } else if (minDist === distToRight) {
          setEdge("right");
        } else if (minDist === distToTop) {
          setEdge("top");
        } else {
          setEdge("bottom");
        }
      };

      const onMouseUp = () => {
        setDragging(false);
        
        // --- SNAPPING LOGIC (STAYS THE SAME) ---
        const { windowWidth, windowHeight, headerHeight, panelWidth, panelHeight } = getLayoutData();
        
        let newPos = { x: 0, y: 0 };

        switch (edge) {
          case "left":
            newPos = { x: 0, y: headerHeight };
            break;
          case "right":
            // Corrected X calculation for snapping to the right edge
            newPos = { x: windowWidth - panelWidth, y: headerHeight }; 
            break;
          case "top":
            // Note: snapping top/bottom assumes full width, so x is 0
            newPos = { x: 0, y: headerHeight };
            break;
          case "bottom":
            newPos = { x: 0, y: windowHeight - panelHeight };
            break;
        }

        // Enforce minY constraint
        if (constraints.minY !== undefined && newPos.y < constraints.minY) {
          newPos.y = constraints.minY;
        }

        // Final snap
        setPos(newPos);

        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      };

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    };

    el.addEventListener("mousedown", onMouseDown);
    return () => el.removeEventListener("mousedown", onMouseDown);
  }, [edge, constraints, pos]); // Added 'pos' to dependency array for correct offset calculation

  return { ref, pos, edge, setPos };
}