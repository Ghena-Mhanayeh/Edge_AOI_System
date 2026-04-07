## Installation & Setup

Um das System auf einer VM oder einem Raspberry Pi (Edge Device) auszuführen, folgen Sie diesen Schritten:

### 1. Repository klonen
```bash
git clone https://github.com/Ghena-Mhanayeh/Edge_AOI_System.git
cd Edge_AOI_System
```

### 2. Virtuelle Umgebung einrichten
Es wird empfohlen, eine virtuelle Umgebung zu verwenden:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux / Mac

# Windows:
venv\Scripts\activate
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 4. Anwendung starten
```bash
python src/main.py
```

### 5. API testen
1. Öffne http://127.0.0.1:8000/docs  
2. Suche den Endpunkt **POST /inspect**  
3. Klicke auf **Try it out**  
4. Wähle ein Bild aus  
5. Klicke auf **Execute**
