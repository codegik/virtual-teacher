import QtQuick

Item {
    id: chat
    signal exit()

    readonly property var statusText: ({
        starting: "Acordando o Pipo…",
        idle: "Pode falar!",
        listening: "Estou ouvindo…",
        thinking: "Pensando…",
        speaking: "",
        error: pipo.error
    })

    Rectangle {
        id: back
        x: 20; y: 20
        width: 120; height: 46
        radius: 12
        color: backArea.containsMouse ? "#303030" : "#262626"
        Text { anchors.centerIn: parent; text: "← Tchau"; color: win.text; font.pixelSize: 18 }
        MouseArea { id: backArea; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: chat.exit() }
    }
    Text {
        anchors { right: parent.right; top: parent.top; margins: 30 }
        text: pipo.kidName
        color: win.muted
        font.pixelSize: 18
    }

    Pipo {
        id: avatar
        width: Math.min(parent.width * 0.6, parent.height * 0.62)
        height: width
        anchors.horizontalCenter: parent.horizontalCenter
        y: parent.height * 0.1
        phase: pipo.phase
        mouth: pipo.mouth
    }

    Text {
        id: status
        anchors { horizontalCenter: parent.horizontalCenter; top: avatar.bottom; topMargin: 28 }
        text: chat.statusText[pipo.phase] ?? ""
        color: pipo.phase === "error" ? win.red : win.text
        font.pixelSize: 30
        // Each new status fades in.
        onTextChanged: appear.restart()
        NumberAnimation { id: appear; target: status; property: "opacity"; from: 0; to: 1; duration: 250 }
    }
}
