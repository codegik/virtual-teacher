import QtQuick

Item {
    id: home
    signal picked(string kidId)

    Column {
        anchors.centerIn: parent
        spacing: 48

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: pipo.ready ? "Quem vai conversar com o Pipo?" : (pipo.error || "Acordando o Pipo…")
            color: pipo.error ? win.red : win.text
            font.pixelSize: 34
            font.bold: true
        }

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 32

            Repeater {
                model: pipo.kids
                delegate: Rectangle {
                    id: tile
                    required property var modelData
                    required property int index
                    width: 210
                    height: 230
                    radius: 22
                    color: area.containsMouse && pipo.ready ? "#262626" : win.panel
                    opacity: pipo.ready ? 1 : 0.45
                    scale: area.pressed ? 0.94 : (area.containsMouse && pipo.ready ? 1.06 : 1)
                    Behavior on scale { SpringAnimation { spring: 4; damping: 0.25 } }
                    Behavior on color { ColorAnimation { duration: 150 } }
                    Behavior on opacity { NumberAnimation { duration: 400 } }

                    // Each tile floats on its own rhythm.
                    transform: Translate {
                        id: float_
                        SequentialAnimation on y {
                            loops: Animation.Infinite
                            NumberAnimation { to: -8; duration: 1600 + tile.index * 300; easing.type: Easing.InOutSine }
                            NumberAnimation { to: 0; duration: 1600 + tile.index * 300; easing.type: Easing.InOutSine }
                        }
                    }

                    Column {
                        anchors.centerIn: parent
                        spacing: 14
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: tile.modelData.emoji
                            font.pixelSize: 76
                            rotation: area.containsMouse && pipo.ready ? 10 : 0
                            Behavior on rotation { SpringAnimation { spring: 3; damping: 0.2 } }
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: tile.modelData.name
                            color: win.text
                            font.pixelSize: 22
                        }
                    }

                    MouseArea {
                        id: area
                        anchors.fill: parent
                        hoverEnabled: true
                        enabled: pipo.ready
                        cursorShape: Qt.PointingHandCursor
                        onClicked: home.picked(tile.modelData.id)
                    }
                }
            }
        }
    }
}
