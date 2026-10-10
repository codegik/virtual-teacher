import QtQuick
import QtQuick.Window

// Soft, low-contrast palette (Matte Black family): amber vs red, never red vs green.
Window {
    id: win
    width: 1100
    height: 760
    visible: true
    color: "#121212"
    title: "Pipo"

    readonly property color text: "#bebebe"
    readonly property color muted: "#8a8a8a"
    readonly property color panel: "#1c1c1c"
    readonly property color accent: "#9fb7d9"
    readonly property color amber: "#d7a65f"
    readonly property color red: "#c97b6f"
    property bool chatting: false

    Home {
        id: home
        anchors.fill: parent
        opacity: win.chatting ? 0 : 1
        scale: win.chatting ? 0.92 : 1
        visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
        Behavior on scale { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
        onPicked: (kidId) => { pipo.start(kidId); win.chatting = true }
    }

    Chat {
        anchors.fill: parent
        opacity: win.chatting ? 1 : 0
        scale: win.chatting ? 1 : 1.08
        visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
        Behavior on scale { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
        onExit: { pipo.stop(); win.chatting = false }
    }
}
