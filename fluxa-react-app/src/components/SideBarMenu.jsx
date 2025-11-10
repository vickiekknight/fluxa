export default function SidebarMenu({ title, options, selected, setSelected, openMenu, setOpenMenu }) {
  const isOpen = openMenu === title;

  const toggleMenu = () => {
    setOpenMenu(isOpen ? null : title);
  };

  return (
    <div>
      <button
        onClick={toggleMenu}
        className={`w-full text-left px-3 py-2 rounded-md transition ${
          isOpen ? "bg-green-600/40 text-green-300" : "hover:bg-green-900/40 text-green-400"
        }`}
      >
        {title} {selected && `: ${selected}`}
      </button>
      {isOpen && (
        <div className="ml-2 mt-1 space-y-1">
          {options.map((opt) => (
            <button
              key={opt}
              onClick={() => setSelected(opt)}
              className={`w-full text-left px-2 py-1 rounded-md hover:bg-green-900/40 transition ${
                selected === opt ? "bg-green-800/60 text-green-300" : "text-green-400"
              }`}
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
