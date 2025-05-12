/*
- HTML : defines content and structure of webpage, like the skeleton of human body.
- CSS : it controls appearance and layout of a webpage, like clothing and accessories.
- JavaScript: Allows dynamic behavior and interactivity to a webpage, like muscle and nerve system.
*/

// script.js
window.addEventListener("DOMContentLoaded", () => {
  const intro = document.getElementById("intro");
  const main = document.getElementById("main");
  const tabs = document.querySelectorAll(".sidebar li");
  const contents = document.querySelectorAll(".tab-content");
  const hashBar = document.getElementById("hash-bar");
  const container = document.querySelector('.container');
  const sidebar = document.getElementById("right-sidebar");
  const button = sidebar.querySelector(".toggle-button");
  button.innerHTML = sidebar.classList.contains("collapsed") ? "&#x25C0;" : "&#x25B6;";

  // Auto-hide intro and show main page
  let hashCount = 1;
  const interval = setInterval(() => {
    hashBar.textContent += ">";
    hashCount++;
    if (hashCount > 12) {
      clearInterval(interval);
      intro.classList.add("hidden");
      main.classList.remove("hidden");
    }
  }, 250);

  // Optional fallback if JS is disabled or hash bar fails
  setTimeout(() => {
    intro.classList.add("hidden");
    main.classList.remove("hidden");
    
  }, 4000);

  // Tab functionality
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      // Deactivate all
      tabs.forEach(t => t.classList.remove("active"));
      contents.forEach(c => {
        c.classList.remove("active");
        c.classList.add("hidden");
      });

      // Activate selected
      const targetId = tab.getAttribute("data-tab");
      const target = document.getElementById(targetId);
      tab.classList.add("active");
      if (target) {
        target.classList.remove("hidden");
        target.classList.add("active");
      }
    });
  });

  // Hyperlink fade-out effect
  document.querySelectorAll('a[target="_blank"]').forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const url = this.href;

      document.body.classList.add('slide-out');

      setTimeout(() => {
        window.open(url, '_blank'); // opens in new tab
        document.body.classList.remove('slide-out'); // optional reset
      }, 1200); // match transition duration
    });
  });

  // load csv data
  loadCSVTable('data/sample.csv', 'csv-table');


});


// Sidebar toggle
function toggleRightSidebar() {
const sidebar = document.getElementById("right-sidebar");
const button = sidebar.querySelector(".toggle-button");

sidebar.classList.toggle("collapsed");

// Flip arrow direction
if (sidebar.classList.contains("collapsed")) {
    button.innerHTML = "&#x25C0;"; // ◀
} else {
    button.innerHTML = "&#x25B6;"; // ▶
}
}

// CSV Table Loader
function loadCSVTable(csvPath, containerId) {
  fetch(csvPath)
    .then(res => res.text())
    .then(text => {
      const rows = text.trim().split('\n').slice(0, 6); // header + 5 rows
      const container = document.getElementById(containerId);
      let html = '<table><thead><tr>';

      // header row
      rows[0].split(',').forEach(col => {
        html += `<th>${col.trim()}</th>`;
      });
      html += '</tr></thead><tbody>';

      // data rows
      rows.slice(1).forEach(row => {
        html += '<tr>';
        row.split(',').forEach(cell => {
          html += `<td>${cell.trim()}</td>`;
        });
        html += '</tr>';
      });

      html += '</tbody></table>';
      container.innerHTML = html;
    })
    .catch(err => {
      document.getElementById(containerId).textContent = 'Failed to load CSV.';
      console.error(err);
    });
}



