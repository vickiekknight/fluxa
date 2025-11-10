import { motion } from "framer-motion";
import { RocketIcon } from "@radix-ui/react-icons";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function AnimatedButton() {
  const navigate = useNavigate();
  const [clicked, setClicked] = useState(false);

  const handleClick = () => {
    setClicked(true);
    setTimeout(() => {
      navigate("/simulation");
    }, 500);
  };

  return (
    <motion.button
      onClick={handleClick}
      animate={{
        boxShadow: clicked
          ? "0 0 30px 10px rgba(0,255,127,0.8), inset 0 0 10px rgba(0,255,127,0.3)"
          : [
              "0 0 5px rgba(0,255,127,0.3), inset 0 0 2px rgba(0,255,127,0.2)",
              "0 0 20px rgba(0,255,127,0.6), inset 0 0 6px rgba(0,255,127,0.3)",
              "0 0 5px rgba(0,255,127,0.3), inset 0 0 2px rgba(0,255,127,0.2)",
            ],
      }}
      transition={{
        duration: 1.2,
        repeat: Infinity,
        repeatType: "loop",
        ease: "easeInOut",
      }}
      whileHover={{
        scale: 1.05,
        boxShadow: "0 0 2em #00ff7faa, inset 0 0 1em #00ff7faa",
      }}
      whileTap={{ scale: 0.97 }}
      className="flex items-center space-x-2 px-6 py-3 rounded-2xl bg-gradient-to-b from-[#003300] via-[#005500] to-[#007700] text-[#00ff7f] border border-[#00ff7f] font-semibold shadow-lg transition-all"
    >
      <RocketIcon className="w-5 h-5 text-[#00ff7f]" />
      <span>Launch Fluxa</span>
    </motion.button>
  );
}