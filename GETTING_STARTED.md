# Getting started: handwritten maths to LaTeX, step by step

This guide is for anyone, with no programming needed. You write a formula (or take a
photo of one) and ink2latex gives you its LaTeX, the text you paste into a LaTeX
document, Overleaf, Typst converters, Notion, Obsidian, MathJax pages and so on.

There are two ways to use it:

- **Way 1, the easiest: open the website.** Nothing to install. Works on a computer,
  tablet or phone.
- **Way 2: run it on your own computer.** For when you want it to run from your own
  files.

Both are the same app. In both, your handwriting and photos stay on your device:
the reading happens inside your browser and nothing is sent to any server.

---

## Way 1: open the website

1. Open **<https://aburousan.github.io/ink2latex/>** in Chrome, Edge, Firefox or
   Safari.
2. Wait for the status line to say **Ready**. The first time it downloads the
   model (about 40 MB), so this can take a minute on a slow connection. After that
   it opens at once, even with no internet.
3. Write your formula in the big box (see [Using the app](#using-the-app) below).

**On a phone or tablet:** in the browser's menu choose *Add to Home Screen* (Safari:
the Share button, then *Add to Home Screen*; Chrome: the ⋮ menu, then *Add to Home
screen* or *Install app*). It then opens like an app and works offline.

---

## Way 2: run it on your own computer

You need two things: **Python 3** (only for its small built-in web server) and a
**browser**.

### Step 1. Install Python (once)

- **Windows:** download Python from <https://www.python.org/downloads/> and run the
  installer. On its first screen, tick **"Add python.exe to PATH"**, then press
  *Install Now*.
- **Mac:** open the **Terminal** app (press ⌘ Space, type `Terminal`, press Return)
  and type `python3 --version`. If it prints a version, you already have it. If a
  window offers to install the "command line developer tools", accept it, or
  install Python from <https://www.python.org/downloads/>.
- **Linux:** Python 3 is almost always installed already.

### Step 2. Download ink2latex (once)

1. Go to <https://github.com/aburousan/ink2latex>.
2. Press the green **Code** button, then **Download ZIP**.
3. Unzip the file `ink2latex-main.zip` (double-click it on a Mac; on Windows,
   right-click it and choose *Extract All…*). You get a folder `ink2latex-main`.

### Step 3. Start it (every time you want to use it)

The app has to be opened through a small local web server. Double-clicking
`index.html` does **not** work: browsers do not let such a page load its model.

**Windows**

1. Open the folder `ink2latex-main`, then the folder `demo` inside it.
2. Click the address bar at the top of the window, type `cmd`, and press Enter.
   A black command window opens in that folder.
3. Type this and press Enter:

   ```
   py -m http.server 8000
   ```

**Mac**

1. Open **Terminal**.
2. Type `cd ` (with a space after it), then drag the `demo` folder from inside
   `ink2latex-main` onto the Terminal window, and press Return.
3. Type this and press Return:

   ```
   python3 -m http.server 8000
   ```

**Linux**

```
cd ~/Downloads/ink2latex-main/demo
python3 -m http.server 8000
```

When it says `Serving HTTP on ... port 8000`, it is running. Leave that window
open.

### Step 4. Open the app

Open your browser and go to **<http://localhost:8000>**. The app opens.

To stop it later, go back to the command window and press **Ctrl+C** (on a Mac
too), or just close that window.

---

## Using the app

### Writing a formula

- Write in the big box with a mouse, a finger or a stylus. Write it the way you
  would on paper, on one line (fractions, powers and roots are fine).
- About half a second after you stop, the LaTeX appears, with the formula typeset
  under it so you can check it.
- **Undo stroke** removes the last stroke (or brings back what the eraser removed).
  **Clear** empties the box. **Copy** copies the LaTeX, ready to paste.
- **Pen** and **Eraser:** with *Eraser* pressed, touch or swipe over a stroke to
  remove it; press *Pen* to write again. The eraser end of a stylus erases without
  pressing anything.
- If you untick **Recognise automatically**, press **Recognise** to read the box.

**For the best results**

- One formula at a time.
- Write a little larger than you would on paper, with clear gaps between symbols.
- Make powers and subscripts clearly smaller and raised or lowered.
- Write fraction bars long enough to cover the top and bottom.

### Reading a photo or a screenshot

1. Give the app a picture in any of these ways:
   - press **Photo** and choose a picture (on a phone or tablet you can take one
     with the camera);
   - **paste** one: copy a picture or take a screenshot (Windows: Win+Shift+S;
     Mac: ⌘ Ctrl Shift 4), then press **Ctrl+V** (⌘V on a Mac) anywhere on the
     page, or press the **Paste** button;
   - **drag** a picture file onto the box.
2. The app shows the strokes it found in the box, so you can see what it is going
   to read, and then reads them.
3. If it picked up stray marks (a shadow, a smudge), remove them with the
   **Eraser**; it reads again.

Photos are harder than writing in the app: a picture has no pen strokes, so the app
has to guess them. To help it:

- Photograph **one formula**, and let it **fill most of the picture**.
- Use a **dark pen on plain white paper** (lined paper is harder).
- Take the photo **straight on**, in **even light**, with no shadow of your hand
  or phone across it.
- Crop the photo to the formula first if you can.

### Using the LaTeX

Paste it into your document between dollar signs, for example in a LaTeX or
Overleaf file:

```latex
$\frac { d y } { d x } = 2 x$
```

The spaces between tokens are harmless; LaTeX ignores them.

---

## When something goes wrong

| What you see | What to do |
| --- | --- |
| **Windows:** `'py' is not recognized` | Python is not installed, or "Add to PATH" was not ticked. Run the Python installer again, choose *Modify*, and tick the PATH option. Then close and reopen the command window. |
| **Mac or Linux:** `command not found: python3` | Install Python from <https://www.python.org/downloads/>. |
| `Address already in use` | Something else uses port 8000. Use another number: `py -m http.server 8080` (or `python3 -m http.server 8080`), then open <http://localhost:8080>. |
| The page opens but nothing is recognised | Wait for the status line to say **Ready** (the first load fetches the model). If you opened `index.html` by double-clicking it, start the server instead (Step 3). |
| A blank or broken page | Use a recent Chrome, Edge, Firefox or Safari. |
| The answer is wrong | Write larger and clearer, with gaps between symbols, and fix the LaTeX by hand. It reads about 6 in 10 formulas of other people's handwriting fully right; most of the rest are close. |
| A symbol it never writes | It knows 243 LaTeX tokens (listed in `demo/models/vocab.json`); anything else it cannot produce. |

---

## Next steps

- [TUTORIAL.md](TUTORIAL.md): use it from Python or the command line, read photos in
  bulk, and teach it your own handwriting.
- [README.md](README.md): how well each model reads, and the full technical details.

ink2latex is by Kazi Abu Rousan. It is free for noncommercial use with credit:
the code under the PolyForm Noncommercial License 1.0.0, the trained model under
CC BY-NC-SA 4.0. It was trained on MathWriting by Google Research (Gervais et al.,
2024).
