function enhanceTables() {
  document.querySelectorAll("table[data-sortable]").forEach((table) => {
    const body = table.tBodies[0];
    if (!body) return;
    table.querySelectorAll("th").forEach((header, index) => {
      header.addEventListener("click", () => {
        const rows = [...body.querySelectorAll("tr")];
        const ascending = header.dataset.order !== "asc";
        rows.sort((a, b) => {
          const left = a.children[index]?.innerText.trim() || "";
          const right = b.children[index]?.innerText.trim() || "";
          const leftNumber = Number(left.replace(/[% ,]/g, ""));
          const rightNumber = Number(right.replace(/[% ,]/g, ""));
          if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber) && left !== "" && right !== "") {
            return (leftNumber - rightNumber) * (ascending ? 1 : -1);
          }
          return left.localeCompare(right, "zh-CN") * (ascending ? 1 : -1);
        });
        rows.forEach((row) => body.appendChild(row));
        table.querySelectorAll("th").forEach((item) => delete item.dataset.order);
        header.dataset.order = ascending ? "asc" : "desc";
      });
    });
  });
}

function mountIcons() {
  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons();
  }
}

function bindFolds() {
  document.querySelectorAll("details.fold").forEach((item) => {
    item.addEventListener("toggle", () => mountIcons());
  });
}

document.addEventListener("DOMContentLoaded", () => {
  enhanceTables();
  bindFolds();
  mountIcons();
});
