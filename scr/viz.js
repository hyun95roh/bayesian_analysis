document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("toggle-btn");
  const checkboxes = document.querySelectorAll("#sidebar-content input[type=checkbox]");
  const plotArea = document.getElementById("plot-area");

  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  // Example visibility
  checkboxes.forEach(cb => {
    cb.addEventListener("change", updatePlots);
  });

  function updatePlots() {
    const selected = Array.from(checkboxes)
      .filter(cb => cb.checked)
      .map(cb => cb.value);

    plotArea.innerHTML = ""; // reset
    selected.forEach(id => {
      const div = document.createElement("div");
      div.id = `plot-${id}`;
      div.style.height = "400px";
      plotArea.appendChild(div);
      plotExample(id, div.id);
    });
  }

  function plotExample(exId, targetId) {
    // Placeholder — link to Python data or JS simulation
    let data = [];
    let layout = {title: ""};
    switch (exId) {
      case "ex1":
        data = [{x:[1,2,3,4,5,6], y:[1/6,1/6,1/6,1/6,1/6,1/6], type:"bar"}];
        layout.title = "Example 1: Dice Probabilities";
        break;
      case "ex2":
        data = [{x:[0,5,10,15,20,30], y:[0.4,0.2,0.15,0.1,0.1,0.05], type:"bar"}];
        layout.title = "Example 2: Bus Waiting Time";
        break;
      default:
        data = [{x:[0,1,2], y:[0.3,0.5,0.2], type:"bar"}];
    }
    Plotly.newPlot(targetId, data, layout);
  }

  updatePlots(); // initial
});
